#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>

typedef struct {
  _Atomic size_t *idx;
  bool *pausep, **paused;
  uint64_t *acc, n, p;
  size_t n_thrds;
  pthread_mutex_t mu, *queue_mu;
  pthread_cond_t cv, *queue_resume;
  bool quit;
} snapshot_st_t;

typedef struct {
  pthread_t ss;
  snapshot_st_t st;
} snapshot_t;

void snapshot_start(snapshot_t *, uint64_t, uint64_t, size_t, pthread_mutex_t *, pthread_cond_t *, bool *, bool **, _Atomic size_t *, uint64_t *);
void snapshot_stop(snapshot_t *restrict);
void snapshot_try_resume(uint64_t, uint64_t, _Atomic size_t *, uint64_t *);
