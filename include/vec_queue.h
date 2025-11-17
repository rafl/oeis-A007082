#pragma once

#include "queue.h"

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>

#define VEC_CHUNK_SIZE (1UL << 15)
#define VEC_Q_CAP 64

// Queue for full necklace vector chunks
typedef struct {
  size_t *vecs, *sizes;
  size_t head, tail, cap, m;
  _Atomic size_t fill;
  bool done, pause;
  pthread_mutex_t mu;
  pthread_cond_t not_empty, not_full, resume;
} vec_queue_t;

typedef void (*vec_queue_resume_cb_t)(void *);
typedef vec_queue_resume_cb_t (*vec_queue_idle_cb_t)(void *);

vec_queue_t *vec_queue_new(size_t, size_t *);

void vec_queue_free(vec_queue_t *);

void vec_queue_push_chunk(vec_queue_t *, const size_t *, size_t);

// Signal that no more chunks will be pushed (called by generator workers when
// done)
void vec_queue_set_done(vec_queue_t *);

size_t vec_queue_pop_chunk(vec_queue_t *, size_t *, vec_queue_idle_cb_t, void *);
