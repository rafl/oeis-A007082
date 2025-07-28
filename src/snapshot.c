#include "snapshot.h"
#include "maths.h"

#include <stdio.h>
#include <inttypes.h>
#include <unistd.h>
#include <limits.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>

#if __APPLE__
#  define _CLOCK CLOCK_REALTIME
#else
#  define _CLOCK CLOCK_MONOTONIC
#endif

static void get_snapshot_path(uint64_t n, uint64_t p, char *buf, size_t len) {
  snprintf(buf, len, ".%"PRIu64".%"PRIu64".ss", n, p);
}

static void snapshot_save(snapshot_st_t *st, size_t idx) {
  char path[PATH_MAX], tmp[PATH_MAX];
  get_snapshot_path(st->n, st->p, path, sizeof(path));
  snprintf(tmp, sizeof(tmp), "%s.tmp", path);

  int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    printf("\nfailed to snapshot (open) %zu %"PRIu64"\n", idx, *st->acc);
    return;
  };

  uint64_t data[2] = { idx, *st->acc };
  if (write(fd, &data, sizeof(data)) != 2) {
    printf("\nfailed to snapshot (write) %zu %"PRIu64"\n", idx, *st->acc);
    close(fd);
    unlink(tmp);
    return;
  }
  fsync(fd);
  close(fd);
  rename(tmp, path);
}

static void *snapshot(void *ud) {
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

    snapshot_save(st, atomic_load_explicit(st->idx, memory_order_seq_cst));

    pthread_mutex_lock(st->queue_mu);
    *st->pausep = false;
    pthread_mutex_unlock(st->queue_mu);
    pthread_cond_signal(st->queue_resume);
  }
  pthread_mutex_unlock(&st->mu);
  return NULL;
}

void snapshot_start(snapshot_t *ss, uint64_t n, uint64_t p, size_t n_thrds, pthread_mutex_t *queue_mu, pthread_cond_t *queue_resume, bool *pausep, bool **paused, _Atomic size_t *idx, uint64_t *acc) {
  snapshot_st_t *st = &ss->st;
  *st = (snapshot_st_t){ .n = n, .p = p, .n_thrds = n_thrds, .queue_mu = queue_mu, .queue_resume = queue_resume, .pausep = pausep, .paused = paused, .idx = idx, .acc = acc };
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
  snapshot_save(st, atomic_load_explicit(st->idx, memory_order_seq_cst));
  pthread_cond_destroy(&st->cv);
  pthread_mutex_destroy(&st->mu);
}

void snapshot_try_resume(uint64_t n, uint64_t p, _Atomic size_t *done, uint64_t *acc) {
  char path[PATH_MAX];
  get_snapshot_path(n, p, path, sizeof(path));
  FILE *f = fopen(path, "r");
  if (!f) {
    if (errno == ENOENT) return;
    perror("fopen");
    abort();
  }

  uint64_t ent[2];
  size_t read = fread(ent, sizeof(uint64_t), 2, f);
  if (read != 2) abort();
  *done = ent[0];
  *acc = ent[1];
  fclose(f);
  //printf("resuming %"PRIu64" from %"PRIu64" with %"PRIu64"\n", p, ent[0], ent[1]);
}
