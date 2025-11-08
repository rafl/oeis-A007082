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
  canon_iter_stage_t stage;
  size_t min_depth;  // Minimum depth for backtracking (0 = root, >0 = resumed)
} canon_iter_t;

void canon_iter_new(canon_iter_t *, size_t, size_t, size_t *);
bool canon_iter_next(canon_iter_t *, size_t *);

size_t canon_iter_size(size_t, size_t);

size_t canon_iter_save(canon_iter_t *it, void *buf, size_t len);
void canon_iter_resume(canon_iter_t *it, size_t m, size_t tot, size_t *scratch,
                       const void *buf, size_t len);

// Frontier builder callback: called with serialized state at depth d
typedef void (*frontier_emit_cb_t)(const void *state, size_t len, void *user);

// Build frontier of tasks at specified depth
void canon_iter_frontier(canon_iter_t *it, size_t depth, void *save_buf,
                        size_t save_buf_len, frontier_emit_cb_t emit_cb,
                        void *user);
