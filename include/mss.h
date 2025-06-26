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
