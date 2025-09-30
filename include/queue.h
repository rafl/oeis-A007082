#pragma once

#include "mss.h"

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>

// This is how many calcs per iteration we do
#define CHUNK (1UL << 17)
#define Q_CAP 16

// shared over all threads - used to pull the next work task

// (there is a "queue thread" that fills up the queue with more tasks)
typedef struct {
  size_t *scratch, *buf, *vecs;
  size_t head, tail, cap, m;
  _Atomic size_t fill;
  bool done, pause;
  canon_iter_t it;
  pthread_mutex_t mu;
  pthread_cond_t not_empty, not_full, resume;
} queue_t;

typedef void (*resume_cb_t)(void *);
typedef resume_cb_t (*idle_cb_t)(void *);

queue_t *queue_new(size_t n, size_t m, const void *iter_st, size_t st_len,
                   size_t *vecs);
void queue_free(queue_t *);
void queue_fill(queue_t *);
size_t queue_pop(queue_t *, size_t *, idle_cb_t, void *);

size_t queue_save(queue_t *it, void *buf, size_t len);
