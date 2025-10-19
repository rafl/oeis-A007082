#pragma once

#include "mss.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
  size_t m, tot, *scratch, t, p, r, u, v, sum;
  bool rs;
  canon_iter_stage_t stage;
} bracelet_iter_t;

void bracelet_iter_new(bracelet_iter_t *, size_t, size_t, size_t *);
bool bracelet_iter_next(bracelet_iter_t *, size_t *);
