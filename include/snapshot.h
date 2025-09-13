#pragma once

#include "queue.h"
#include "source_process.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>

typedef struct {
  _Atomic size_t *idx;
  bool **paused;
  queue_t *q;
  uint64_t *acc, n, p;
  size_t n_thrds;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  bool quit;
  process_mode_t mode;
} snapshot_st_t;

typedef struct {
  pthread_t ss;
  snapshot_st_t st;
} snapshot_t;

void snapshot_start(snapshot_t *, process_mode_t, uint64_t, uint64_t, size_t, queue_t *, bool **, _Atomic size_t *, uint64_t *);
void snapshot_stop(snapshot_t *restrict);
void snapshot_try_resume(process_mode_t, uint64_t n, uint64_t p, _Atomic size_t *done, uint64_t *acc, void *iter_st, size_t *st_len);
