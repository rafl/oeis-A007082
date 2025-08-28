#pragma once

#include <stdint.h>

// Source of residues modulo some prime
typedef struct source_St {
  // do work
  int  (*next)(struct source_St *, uint64_t *res, uint64_t *p);
  // destructor
  void (*destroy)(struct source_St *);
  // virtualization - store arb state
  void *state;
} source_t;
