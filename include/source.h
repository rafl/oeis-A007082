#pragma once

#include <stdint.h>

typedef struct source_St {
  int  (*next)(struct source_St *, uint64_t *res, uint64_t *p);
  void (*destroy)(struct source_St *);
  void *state;
} source_t;
