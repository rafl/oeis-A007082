#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef enum {
  ITER_STAGE_DESCEND,
  ITER_STAGE_LOOP,
  ITER_STAGE_BACKTRACK,
} canon_iter_stage_t;

typedef struct {
  size_t m, tot, *scratch, t, p, sum;
  canon_iter_stage_t stage;
} canon_iter_t;

void canon_iter_new(canon_iter_t *, size_t, size_t, size_t *);
bool canon_iter_next(canon_iter_t *, size_t *);
size_t canon_iter_size(size_t, size_t);
