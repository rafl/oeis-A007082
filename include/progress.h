#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdatomic.h>

#ifdef __cplusplus 
#include <atomic>
#define Atomic std::atomic<size_t>
#else
#define Atomic _Atomic size_t
#endif

typedef struct {
  Atomic * done;
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

void progress_start(progress_t *, uint64_t, Atomic *, size_t);
void progress_stop(progress_t *restrict);
