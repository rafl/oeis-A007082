#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef enum {
  ITER_STAGE_DESCEND,
  ITER_STAGE_LOOP,
  ITER_STAGE_BACKTRACK,
} canon_iter_stage_t;

// This stuff is the source of each unique set of coefficients
typedef struct {
  size_t m, tot, *scratch, t, p, sum;
  size_t depth;       // target depth: emit result when t == depth + 1
  size_t start_depth; // backtracking stops when t <= start_depth
  canon_iter_stage_t stage;
} canon_iter_t;

void canon_iter_new(canon_iter_t *, size_t m, size_t tot, size_t *scratch,
                    size_t depth);
bool canon_iter_next(canon_iter_t *, size_t *);

void canon_iter_from_prefix(canon_iter_t *it, size_t m, size_t tot,
                            size_t *scratch, const size_t *prefix,
                            size_t prefix_depth);

size_t canon_iter_depth_for(size_t m);

size_t canon_iter_size(size_t, size_t);

size_t canon_iter_save(canon_iter_t *it, void *buf, size_t len);
void canon_iter_resume(canon_iter_t *it, size_t m, size_t tot, size_t *scratch,
                       const void *buf, size_t len);
