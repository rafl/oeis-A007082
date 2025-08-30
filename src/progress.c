#include "progress.h"
#include "queue.h"

#include <stdio.h>
#include <inttypes.h>

#if __APPLE__
#  define _CLOCK CLOCK_REALTIME
#else
#  define _CLOCK CLOCK_MONOTONIC
#endif

#define PROG_INT 1
#define PROG_RATE_TAU 60.0

static void *progress(void *_ud) {
  progress_st_t *ud = _ud;
  size_t tot = ud->tot;
  _Atomic size_t *done = ud->done;
  struct timespec last = ud->start;
  double rate_avg = 0;
  // technically there's a race? it's fine!
  size_t prev = atomic_load_explicit(done, memory_order_relaxed);

  pthread_mutex_lock(&ud->mu);
  while (!ud->quit) {
    struct timespec now;
    clock_gettime(_CLOCK, &now);
    size_t d = atomic_load_explicit(done, memory_order_relaxed);
    double dt = (now.tv_sec - last.tv_sec) + (now.tv_nsec - last.tv_nsec)*1e-9;
    double rate = (double)(d-prev) / dt;
    if (rate_avg == 0) rate_avg = rate;
    else rate_avg += (dt / (PROG_RATE_TAU+dt)) * (rate-rate_avg);
    prev = d;
    last = now;
    double pct = 100.0 * d / tot;
    double elapsed = (now.tv_sec - ud->start.tv_sec) + (now.tv_nsec - ud->start.tv_nsec)*1e-9;
    double eta = (d && d < tot && rate_avg > 0) ? (tot-d) / rate_avg : 0.0;
    int eh = elapsed / 3600, es = (int)elapsed % 60, em = ((int)elapsed / 60) % 60;
    int th = (eta / 3600), ts = (int)eta % 60, tm = ((int)eta / 60) % 60;
    fprintf(stderr, "\r%5.2f%% | %02d:%02d:%02d | %.2fM/s | %3.0f%% | ETA %02d:%02d:%02d (%"PRIu64")",
            pct, eh, em, es, rate_avg/1e6, ((float)atomic_load_explicit(ud->q_fill, memory_order_relaxed)) * 100 / (CHUNK*Q_CAP), th, tm, ts, ud->p);
    if (d >= tot) break;
    now.tv_sec += PROG_INT;
    pthread_cond_timedwait(&ud->cv, &ud->mu, &now);
  }
  pthread_mutex_unlock(&ud->mu);
  fprintf(stderr, "\n");
  return NULL;
}

void progress_start(progress_t *p, uint64_t prime, _Atomic size_t *done, size_t tot, _Atomic size_t *q_fill) {
  progress_st_t *st = &p->st;
  *st = (progress_st_t){ .p = prime, .done = done, .tot = tot, .quit = false, .q_fill = q_fill };
  pthread_mutex_init(&st->mu, NULL);
  clock_gettime(_CLOCK, &st->start);
  pthread_condattr_t ca;
  pthread_condattr_init(&ca);
#if !__APPLE__
  pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
#endif
  pthread_cond_init(&st->cv, &ca);
  pthread_condattr_destroy(&ca);

  pthread_create(&p->prog, NULL, progress, st);
}

void progress_stop(progress_t *restrict p) {
  progress_st_t *st = &p->st;
  pthread_mutex_lock(&st->mu);
  st->quit = true;
  pthread_cond_signal(&st->cv);
  pthread_mutex_unlock(&st->mu);
  pthread_join(p->prog, NULL);
  pthread_cond_destroy(&st->cv);
  pthread_mutex_destroy(&st->mu);
}
