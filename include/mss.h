#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef struct {
  size_t n, tot, lvl, *vec, *scratch;
  bool fin;
} mss_iter_t;

void mss_iter_new(mss_iter_t *const, size_t, size_t, size_t *, size_t *);
void mss_iter_init_at(mss_iter_t *const, size_t, size_t, size_t *, size_t *);
size_t mss_iter_size(size_t, size_t);
bool mss_iter(mss_iter_t *restrict);

typedef enum {
  ITER_STAGE_DESCEND,
  ITER_STAGE_LOOP,
  ITER_STAGE_BACKTRACK,
} canon_iter_stage_t;

typedef struct {
  size_t m, tot, *vec, *scratch, t, p, sum;
  canon_iter_stage_t stage;
} canon_iter_t;

void canon_iter_new(canon_iter_t *, size_t, size_t, size_t *, size_t *);
bool canon_iter_next(canon_iter_t *);
