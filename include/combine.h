#pragma once

#include <stdbool.h>
#include <stdint.h>

// TODO: make opaue and provide accessors, avoiding exposure of gmp?
#include <gmp.h>
typedef struct { mpz_t X, M; } comb_ctx_t;

comb_ctx_t *comb_ctx_new();
void comb_ctx_free(comb_ctx_t *);
bool comb_ctx_add(comb_ctx_t *ctx, uint64_t res, uint64_t p);
