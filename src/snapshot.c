#include "snapshot.h"
#include "maths.h"

#include <stdio.h>
#include <inttypes.h>
#include <unistd.h>

#if __APPLE__
#  define _CLOCK CLOCK_REALTIME
#else
#  define _CLOCK CLOCK_MONOTONIC
#endif

static void snapshot_save (snapshot_st_t *st, size_t idx) {
  printf("\n\nSS: %"PRIu64" at %zu\n\n", *st->acc, idx);
}

static void *snapshot (void *ud) {
  snapshot_st_t *st = ud;
  pthread_mutex_lock(&st->mu);
  while (!st->quit) {
    struct timespec now;
    clock_gettime(_CLOCK, &now);
    now.tv_sec += 60;
    pthread_cond_timedwait(&st->cv, &st->mu, &now);
    if (st->quit) break;

    pthread_mutex_lock(st->queue_mu);
    *st->pausep = true;
    pthread_mutex_unlock(st->queue_mu);

    bool all_paused;
    do {
      atomic_thread_fence(memory_order_seq_cst);
      all_paused = true;
      for (size_t i = 0; i < st->n_thrds; ++i) {
        if (!*st->paused[i]) {
          all_paused = false;
          break;
        }
      }
      if (!all_paused) usleep(1000);
    } while (!all_paused);

    for (size_t i = 0; i < st->n_thrds; ++i) {
      uint64_t *t_acc = *st->accs[i];
      *st->acc = add_mod_u64(*st->acc, *t_acc, st->p);
      *t_acc = 0;
    }
    snapshot_save(st, atomic_load_explicit(st->idx, memory_order_seq_cst));

    pthread_mutex_lock(st->queue_mu);
    *st->pausep = false;
    pthread_mutex_unlock(st->queue_mu);
    pthread_cond_signal(st->queue_resume);
  }
  pthread_mutex_unlock(&st->mu);
  return NULL;
}

void snapshot_start(snapshot_t *ss, uint64_t p, size_t n_thrds, pthread_mutex_t *queue_mu, pthread_cond_t *queue_resume, bool *pausep, bool **paused, _Atomic size_t *idx, uint64_t *acc, uint64_t ***accs) {
  snapshot_st_t *st = &ss->st;
  *st = (snapshot_st_t){ .p = p, .n_thrds = n_thrds, .queue_mu = queue_mu, .queue_resume = queue_resume, .pausep = pausep, .paused = paused, .idx = idx, .acc = acc, .accs = accs };
  pthread_mutex_init(&st->mu, NULL);
  pthread_condattr_t ca;
  pthread_condattr_init(&ca);
#if !__APPLE__
  pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
#endif
  pthread_cond_init(&st->cv, &ca);
  pthread_condattr_destroy(&ca);

  pthread_create(&ss->ss, NULL, snapshot, st);
}

void snapshot_stop(snapshot_t *restrict ss) {
  snapshot_st_t *st = &ss->st;
  pthread_mutex_lock(&st->mu);
  st->quit = true;
  pthread_cond_signal(&st->cv);
  pthread_mutex_unlock(&st->mu);
  pthread_join(ss->ss, NULL);
  pthread_cond_destroy(&st->cv);
  pthread_mutex_destroy(&st->mu);
}
