#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>

typedef struct {
  _Atomic size_t *done, *q_fill;
  size_t tot;
  bool quit;
  pthread_cond_t cv;
  pthread_mutex_t mu;
  struct timespec start;
  uint64_t p;
} progress_st_t;

typedef struct {
  progress_st_t st;
  pthread_t prog;
} progress_t;

void progress_start(progress_t *, uint64_t, _Atomic size_t *, size_t, _Atomic size_t *);
void progress_stop(progress_t *restrict);
