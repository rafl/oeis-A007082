#pragma once

#include "mss.h"

#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

#define CHUNK (1UL<<17)

typedef struct {
  size_t *scratch, *buf, *vecs;
  size_t head, tail, cap, fill, m;
  bool done;
  canon_iter_t it;
  pthread_mutex_t mu;
  pthread_cond_t not_empty;
  pthread_cond_t not_full;
} queue_t;

queue_t *queue_new(size_t, size_t);
void queue_free(queue_t *);
void queue_fill(queue_t *);
size_t queue_pop(queue_t *, size_t *, bool *);
