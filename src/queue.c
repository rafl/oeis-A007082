#include "queue.h"
#include "debug.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#include <stdio.h>

queue_t *queue_new(size_t n, size_t m, size_t from, size_t *vecs) {
  queue_t *q = malloc(sizeof(queue_t));
  assert(q);
  q->head = q->tail = q->fill = 0;
  q->cap = Q_CAP * CHUNK;
  q->m = m;
  q->done = q->pause = false;
  pthread_mutex_init(&q->mu, NULL);
  pthread_cond_init(&q->not_empty, NULL);
  pthread_cond_init(&q->not_full, NULL);
  pthread_cond_init(&q->resume, NULL);
  q->scratch = malloc((m+1)*sizeof(size_t));
  assert(q->scratch);
  canon_iter_new(&q->it, m, n, q->scratch);
  q->vecs = vecs;
  q->buf = &vecs[CHUNK*m];

  while (from--) canon_iter_next(&q->it, q->vecs);

  return q;
}

void queue_free(queue_t *q) {
  free(q->scratch);
}

static inline void queue_push(queue_t *restrict q, const size_t *vecs, size_t n_vec) {
  size_t m = q->m;
  pthread_mutex_lock(&q->mu);

  if (q->pause) {
    pthread_cond_wait(&q->resume, &q->mu);
  }

  while (q->fill + n_vec > q->cap)
    pthread_cond_wait(&q->not_full, &q->mu);

  size_t spc = q->cap - q->tail;
  size_t fst = (n_vec <= spc) ? n_vec : spc;

  memcpy(&q->buf[q->tail*m], vecs, fst*m*sizeof(size_t));

  if (fst < n_vec)
    memcpy(q->buf, &vecs[fst*m], (n_vec - fst)*m*sizeof(size_t));

  q->tail = (q->tail + n_vec) % q->cap;
  q->fill += n_vec;

  pthread_cond_signal(&q->not_empty);
  pthread_mutex_unlock(&q->mu);
}

size_t queue_pop(queue_t *q, size_t *out, bool *idlep) {
  size_t m = q->m;
  pthread_mutex_lock(&q->mu);

  while (q->fill == 0 && !q->done) {
    atomic_thread_fence(memory_order_seq_cst);
    *idlep = true;
    pthread_cond_wait(&q->not_empty, &q->mu);
  }

  if (q->fill == 0 && q->done) {
    pthread_mutex_unlock(&q->mu);
    return 0;
  }

  *idlep = false;
  size_t n_vec = q->fill < CHUNK ? q->fill : CHUNK;

  size_t spc = q->cap - q->head;
  size_t fst = n_vec <= spc ? n_vec : spc;

  memcpy(out, &q->buf[q->head*m], fst*m*sizeof(size_t));
  if (fst < n_vec)
    memcpy(out + fst*m, q->buf, (n_vec - fst)*m*sizeof(size_t));

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
      if (!canon_iter_next(&q->it, &q->vecs[n_vec*q->m]))
        break;
    }

    queue_push(q, q->vecs, n_vec);

    if (n_vec < CHUNK) break;
  }

  pthread_mutex_lock(&q->mu);
  q->done = true;
  pthread_cond_broadcast(&q->not_empty);
  pthread_mutex_unlock(&q->mu);
}
