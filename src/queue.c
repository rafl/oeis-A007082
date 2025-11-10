#include "queue.h"
#include "debug.h"

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

queue_t *queue_new(size_t n_args, size_t m, size_t prefix_depth,
                   const void *iter_st, size_t st_len, size_t *vecs) {
  queue_t *q = malloc(sizeof(queue_t));
  assert(q);
  q->head = q->tail = q->fill = 0;
  q->cap = Q_CAP * CHUNK;
  q->m = m;
  q->prefix_depth = prefix_depth;
  q->done = q->pause = false;
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
  pthread_cond_init(&q->resume, NULL);
  q->scratch = malloc((m + 1) * sizeof(size_t));
  assert(q->scratch);
  if (st_len)
    canon_iter_resume(&q->it, m, n_args, q->scratch, iter_st, st_len);
  else
    canon_iter_new(&q->it, m, n_args, q->scratch, prefix_depth);
  q->vecs = vecs;
  q->buf = &vecs[CHUNK * prefix_depth];

  return q;
}

size_t queue_save(queue_t *q, void *buf, size_t len) {
  return canon_iter_save(&q->it, buf, len);
}

void queue_free(queue_t *q) { free(q->scratch); }

static inline void queue_push(queue_t *restrict q, const size_t *vecs,
                              size_t n_vec) {
  size_t stride = q->prefix_depth;
  pthread_mutex_lock(&q->mu);

  while (q->fill + n_vec > q->cap)
    pthread_cond_wait(&q->not_full, &q->mu);

  size_t spc = q->cap - q->tail;
  size_t fst = (n_vec <= spc) ? n_vec : spc;

  memcpy(&q->buf[q->tail * stride], vecs, fst * stride * sizeof(size_t));

  if (fst < n_vec)
    memcpy(q->buf, &vecs[fst * stride], (n_vec - fst) * stride * sizeof(size_t));

  q->tail = (q->tail + n_vec) % q->cap;
  q->fill += n_vec;

  pthread_cond_signal(&q->not_empty);

  if (q->pause)
    pthread_cond_wait(&q->resume, &q->mu);

  pthread_mutex_unlock(&q->mu);
}

size_t queue_pop(queue_t *q, size_t *out, idle_cb_t onidle, void *ud) {
  size_t stride = q->prefix_depth;
  pthread_mutex_lock(&q->mu);

  while (q->fill == 0 && !q->done) {
    resume_cb_t resume = onidle(ud);
    pthread_cond_wait(&q->not_empty, &q->mu);
    resume(ud);
  }

  if (q->fill == 0 && q->done) {
    pthread_mutex_unlock(&q->mu);
    return 0;
  }

  size_t n_vec = q->fill < CHUNK ? q->fill : CHUNK;

  size_t spc = q->cap - q->head;
  size_t fst = n_vec <= spc ? n_vec : spc;

  memcpy(out, &q->buf[q->head * stride], fst * stride * sizeof(size_t));
  if (fst < n_vec)
    memcpy(out + fst * stride, q->buf, (n_vec - fst) * stride * sizeof(size_t));

  q->head = (q->head + n_vec) % q->cap;
  q->fill -= n_vec;

  pthread_cond_signal(&q->not_full);
  pthread_mutex_unlock(&q->mu);

  return n_vec;
}

void queue_fill(queue_t *restrict q) {
  for (;;) {
    size_t n_vec = 0;
    for (; n_vec < CHUNK; ++n_vec) {
      if (!canon_iter_next(&q->it, &q->vecs[n_vec * q->prefix_depth]))
        break;
    }

    queue_push(q, q->vecs, n_vec);

    if (n_vec < CHUNK)
      break;
  }

  pthread_mutex_lock(&q->mu);
  q->done = true;
  pthread_cond_broadcast(&q->not_empty);
  pthread_mutex_unlock(&q->mu);
}

// ============================================================================
// Frontier-based parallel generation
// ============================================================================

// Task structure for frontier-based parallelism
typedef struct {
  uint8_t *state;
  size_t len;
} task_t;

// Task queue for distributing work
typedef struct {
  task_t *tasks;
  size_t capacity;
  _Atomic size_t count;
  _Atomic size_t next_task;
  pthread_mutex_t mu;
} task_queue_t;

static task_queue_t *task_queue_new(size_t capacity) {
  task_queue_t *tq = malloc(sizeof(task_queue_t));
  tq->tasks = malloc(capacity * sizeof(task_t));
  tq->capacity = capacity;
  tq->count = 0;
  tq->next_task = 0;
  pthread_mutex_init(&tq->mu, NULL);
  return tq;
}

static void task_queue_free(task_queue_t *tq) {
  for (size_t i = 0; i < tq->count; ++i) {
    free(tq->tasks[i].state);
  }
  free(tq->tasks);
  free(tq);
}

// Frontier callback: add task to queue
static void frontier_emit_task(const void *state, size_t len, void *user) {
  task_queue_t *tq = user;
  pthread_mutex_lock(&tq->mu);

  // Grow if needed
  if (tq->count >= tq->capacity) {
    size_t new_cap = tq->capacity * 2;
    tq->tasks = realloc(tq->tasks, new_cap * sizeof(task_t));
    assert(tq->tasks);
    tq->capacity = new_cap;
  }

  task_t *t = &tq->tasks[tq->count++];
  t->len = len;
  t->state = malloc(len);
  memcpy(t->state, state, len);


  pthread_mutex_unlock(&tq->mu);
}

// Worker thread: resume from tasks and enumerate
typedef struct {
  task_queue_t *tq;
  queue_t *q;
  size_t m, tot;
  _Atomic size_t *total_count;
} worker_ctx_t;

static void *worker_thread(void *arg) {
  worker_ctx_t *ctx = arg;
  size_t *scratch = malloc((ctx->m + 1) * sizeof(size_t));
  memset(scratch, 0, (ctx->m + 1) * sizeof(size_t));
  size_t prefix_depth = ctx->q->prefix_depth;
  size_t *vecs = malloc(CHUNK * prefix_depth * sizeof(size_t));

  for (;;) {
    // Get next task atomically
    size_t task_idx = atomic_fetch_add(&ctx->tq->next_task, 1);
    if (task_idx >= ctx->tq->count)
      break;

    task_t *task = &ctx->tq->tasks[task_idx];

    // Resume iterator from saved state
    canon_iter_t it;
    canon_iter_resume(&it, ctx->m, ctx->tot, scratch, task->state, task->len);

    // Enumerate from this state (generating prefixes)
    for (;;) {
      size_t n_vec = 0;
      for (; n_vec < CHUNK; ++n_vec) {
        if (!canon_iter_next(&it, &vecs[n_vec * prefix_depth]))
          break;
      }

      if (n_vec > 0) {
        atomic_fetch_add(ctx->total_count, n_vec);
        queue_push(ctx->q, vecs, n_vec);
      }

      if (n_vec < CHUNK)
        break;
    }
  }

  free(vecs);
  free(scratch);
  return NULL;
}

void queue_fill_parallel(queue_t *restrict q, size_t n_threads, size_t depth) {
  if (n_threads <= 1 || depth == 0 || depth >= q->m) {
    queue_fill(q);
    return;
  }

  // Build frontier at specified depth
  size_t state_size = (6 + q->m + 1) * sizeof(uint64_t);  // Updated for depth/start_depth fields
  void *save_buf = malloc(state_size);

  // Start with modest capacity, will grow as needed
  size_t initial_cap = 1024;
  task_queue_t *tq = task_queue_new(initial_cap);

  // Allocate separate scratch for frontier to avoid interference
  size_t *frontier_scratch = malloc((q->m + 1) * sizeof(size_t));
  memset(frontier_scratch, 0, (q->m + 1) * sizeof(size_t));

  canon_iter_t frontier_it;
  canon_iter_new(&frontier_it, q->m, q->it.tot, frontier_scratch, q->prefix_depth);
  canon_iter_frontier(&frontier_it, depth, save_buf, state_size,
                     frontier_emit_task, tq);

  free(frontier_scratch);
  free(save_buf);


  // Launch workers
  pthread_t *threads = malloc(n_threads * sizeof(pthread_t));
  worker_ctx_t *workers = malloc(n_threads * sizeof(worker_ctx_t));
  _Atomic size_t total_count = 0;

  for (size_t i = 0; i < n_threads; ++i) {
    workers[i].tq = tq;
    workers[i].q = q;
    workers[i].m = q->m;
    workers[i].tot = q->it.tot;
    workers[i].total_count = &total_count;
    pthread_create(&threads[i], NULL, worker_thread, &workers[i]);
  }

  // Wait for completion
  for (size_t i = 0; i < n_threads; ++i) {
    pthread_join(threads[i], NULL);
  }


  free(workers);
  free(threads);
  task_queue_free(tq);

  pthread_mutex_lock(&q->mu);
  q->done = true;
  pthread_cond_broadcast(&q->not_empty);
  pthread_mutex_unlock(&q->mu);
}
