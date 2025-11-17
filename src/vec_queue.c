#include "vec_queue.h"

#include <assert.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

vec_queue_t *vec_queue_new(size_t m, size_t *vecs) {
  vec_queue_t *vq = malloc(sizeof(vec_queue_t));
  assert(vq);

  size_t *sizes = malloc(VEC_Q_CAP * sizeof(size_t));
  assert(sizes);

  *vq = (vec_queue_t){
      .vecs = vecs,
      .sizes = sizes,
      .head = 0,
      .tail = 0,
      .cap = VEC_Q_CAP,
      .m = m,
      .fill = 0,
      .done = false,
      .pause = false,
      .mu = PTHREAD_MUTEX_INITIALIZER,
      .not_empty = PTHREAD_COND_INITIALIZER,
      .not_full = PTHREAD_COND_INITIALIZER,
      .resume = PTHREAD_COND_INITIALIZER,
  };

  return vq;
}

void vec_queue_free(vec_queue_t *vq) {
  pthread_mutex_destroy(&vq->mu);
  pthread_cond_destroy(&vq->not_empty);
  pthread_cond_destroy(&vq->not_full);
  pthread_cond_destroy(&vq->resume);
  free(vq->sizes);
  free(vq);
}

void vec_queue_push_chunk(vec_queue_t *vq, const size_t *chunk, size_t n_vecs) {
  assert(n_vecs > 0 && n_vecs <= VEC_CHUNK_SIZE);

  pthread_mutex_lock(&vq->mu);
  while (vq->fill >= vq->cap) {
    pthread_cond_wait(&vq->not_full, &vq->mu);
  }

  size_t *dst = &vq->vecs[vq->tail * VEC_CHUNK_SIZE * vq->m];
  memcpy(dst, chunk, n_vecs * vq->m * sizeof(size_t));
  vq->sizes[vq->tail] = n_vecs;
  vq->tail = (vq->tail + 1) % vq->cap;
  atomic_fetch_add(&vq->fill, 1);

  pthread_cond_signal(&vq->not_empty);
  pthread_mutex_unlock(&vq->mu);
}

void vec_queue_set_done(vec_queue_t *vq) {
  pthread_mutex_lock(&vq->mu);
  vq->done = true;
  pthread_cond_broadcast(&vq->not_empty); // Wake all waiting consumers
  pthread_mutex_unlock(&vq->mu);
}

size_t vec_queue_pop_chunk(vec_queue_t *vq, size_t *dst, vec_queue_idle_cb_t onidle,
                           void *ud) {
  pthread_mutex_lock(&vq->mu);
  while (vq->fill == 0 && !vq->done) {
    vec_queue_resume_cb_t resume = onidle(ud);
    pthread_cond_wait(&vq->not_empty, &vq->mu);
    resume(ud);
  }

  if (vq->fill == 0 && vq->done) {
    pthread_mutex_unlock(&vq->mu);
    return 0;
  }

  size_t n_vecs = vq->sizes[vq->head];
  size_t *src = &vq->vecs[vq->head * VEC_CHUNK_SIZE * vq->m];

  memcpy(dst, src, n_vecs * vq->m * sizeof(size_t));
  vq->head = (vq->head + 1) % vq->cap;
  atomic_fetch_sub(&vq->fill, 1);

  pthread_cond_signal(&vq->not_full);
  pthread_mutex_unlock(&vq->mu);

  return n_vecs;
}
