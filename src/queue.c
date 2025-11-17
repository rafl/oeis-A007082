#include "queue.h"
#include "debug.h"

#include <stdatomic.h>
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

void queue_free(queue_t *q) {
  free(q->scratch);
  free(q);
}

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
    memcpy(q->buf, &vecs[fst * stride],
           (n_vec - fst) * stride * sizeof(size_t));

  q->tail = (q->tail + n_vec) % q->cap;
  q->fill += n_vec;

  pthread_cond_signal(&q->not_empty);

  if (q->pause)
    pthread_cond_wait(&q->resume, &q->mu);

  pthread_mutex_unlock(&q->mu);
}

size_t queue_pop(queue_t *q, size_t *out, queue_idle_cb_t onidle, void *ud) {
  size_t stride = q->prefix_depth;
  pthread_mutex_lock(&q->mu);

  while (q->fill == 0 && !q->done) {
    onidle(ud);
    pthread_cond_wait(&q->not_empty, &q->mu);
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
