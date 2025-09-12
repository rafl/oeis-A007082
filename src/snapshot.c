#include "debug.h"
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

const char * const infix[3] = {
  "", // MODE_REG
  ".jackoff", // MODE_JACKOFF
  ".jackest", // MODE_JACKEST
};

static void get_snapshot_path(process_mode_t mode, uint64_t n, uint64_t p, char *buf, size_t len) {
  assert(mode <= MODE_JACKEST);
  snprintf(buf, len, ".%"PRIu64".%"PRIu64"%s.ss", n, p, infix[mode]);
}

static void snapshot_save(snapshot_st_t *st, size_t idx) {
  char path[PATH_MAX-4], tmp[PATH_MAX];
  get_snapshot_path(st->mode, st->n, st->p, path, sizeof(path));
  snprintf(tmp, sizeof(tmp), "%s.tmp", path);

  int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    printf("\nfailed to snapshot (open) %zu %"PRIu64"\n", idx, *st->acc);
    return;
  };

  uint64_t iter_st[4+st->q->m+1];
  size_t st_len = queue_save(st->q, iter_st, (4+st->q->m+1)*sizeof(uint64_t));
  uint64_t data[3] = { idx, *st->acc, st_len };
  if (write(fd, &data, sizeof(data)) != 3*sizeof(uint64_t)) {
    printf("\nfailed to snapshot (write) %zu %"PRIu64"\n", idx, *st->acc);
    close(fd);
    unlink(tmp);
    return;
  }
  if (write(fd, iter_st, st_len) != (int)st_len) {
    printf("\nfailed to snapshot (write iter) %zu %"PRIu64"\n", idx, *st->acc);
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
    now.tv_sec += 60*30;
    pthread_cond_timedwait(&st->cv, &st->mu, &now);
    if (st->quit) break;

    pthread_mutex_lock(&st->q->mu);
    st->q->pause = true;
    pthread_mutex_unlock(&st->q->mu);

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

    pthread_mutex_lock(&st->q->mu);
    st->q->pause = false;
    pthread_mutex_unlock(&st->q->mu);
    pthread_cond_signal(&st->q->resume);
  }
  pthread_mutex_unlock(&st->mu);
  return NULL;
}

void snapshot_start(snapshot_t *ss, process_mode_t mode, uint64_t n, uint64_t p, size_t n_thrds, queue_t *q, bool **paused, _Atomic size_t *idx, uint64_t *acc) {
  snapshot_st_t *st = &ss->st;
  *st = (snapshot_st_t){ .mode = mode, .n = n, .p = p, .n_thrds = n_thrds, .q = q, .paused = paused, .idx = idx, .acc = acc };
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

void snapshot_try_resume(process_mode_t mode, uint64_t n, uint64_t p, _Atomic size_t *done, uint64_t *acc, void *iter_st, size_t *st_len) {
  char path[PATH_MAX];
  get_snapshot_path(mode, n, p, path, sizeof(path));
  *st_len = 0;
  FILE *f = fopen(path, "r");
  if (!f) {
    if (errno == ENOENT) return;
    perror("fopen");
    abort();
  }

  uint64_t ent[3];
  size_t read = fread(ent, sizeof(uint64_t), 3, f);
  if (read != 3) abort();
  *done = ent[0];
  *acc = ent[1];
  *st_len = ent[2];
  read = fread(iter_st, 1, ent[2], f);
  if (read != ent[2]) abort();
  fclose(f);
}
