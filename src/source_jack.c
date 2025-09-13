#include "debug.h"
#include "source_jack.h"
#include "maths.h"

#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

typedef struct {
  source_t *off, *est;
  bool quiet;
} proc_state_t;

static int proc_next(source_t *self, uint64_t *res, uint64_t *p_ret) {
  proc_state_t *st = self->state;
  uint64_t off, offp;
  bool ok = st->off->next(st->off, &off, &offp);
  if (!ok) return ok;
  if (!st->quiet) printf("off: %"PRIu64" %% %"PRIu64"\n", off, offp);
  uint64_t est, estp;
  VERIFY(st->est->next(st->est, &est, &estp));
  assert(offp == estp);
  if (!st->quiet) printf("est: %"PRIu64" %% %"PRIu64"\n", est, estp);
  *p_ret = offp;
  *res = sub_mod_u64(est, off, offp);
  return ok;
}

static void proc_destroy(source_t *self) {
  proc_state_t *st = self->state;
  st->off->destroy(st->off);
  st->est->destroy(st->est);
  free(self);
}

source_t *source_jack_new(process_mode_t, uint64_t n, uint64_t m_id, bool quiet, bool snapshot, size_t *vecss) {
  assert(!vecss);
  source_t *off = source_process_new(PROC_MODE_JACK_OFFSET, n, m_id, quiet, snapshot, NULL);
  source_t *est = source_process_new(PROC_MODE_JACKEST, n, m_id, quiet, snapshot, source_process_vecss(off));

  proc_state_t *st = malloc(sizeof(*st));
  assert(st);
  *st = (proc_state_t){ .off = off, .est = est, .quiet = quiet };

  source_t *src = malloc(sizeof *src);
  assert(src);
  *src = (source_t){ .next = proc_next, .destroy = proc_destroy, .state = st };
  return src;
}
