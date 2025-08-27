#include "debug.h"
#include "combine.h"

#include <stdlib.h>
#include <gmp.h>

comb_ctx_t *comb_ctx_new() {
  comb_ctx_t *ctx = (comb_ctx_t *)malloc(sizeof(comb_ctx_t));
  mpz_inits(ctx->X, ctx->M, NULL);
  mpz_set_ui(ctx->X, 0);
  mpz_set_ui(ctx->M, 1);
  return ctx;
}

void comb_ctx_free(comb_ctx_t *ctx) {
  mpz_clears(ctx->X, ctx->M, NULL);
  free(ctx);
}

bool comb_ctx_add(comb_ctx_t *ctx, uint64_t res, uint64_t p) {
  mpz_t u, inv, mz, rz, Xp;
  mpz_inits(u, inv, mz, rz, Xp, NULL);

  mpz_set(Xp, ctx->X);

  mpz_set_ui(rz, res);
  mpz_set_ui(mz, p);

  mpz_mod(u, ctx->X, mz);
  mpz_sub(u, rz, u);
  mpz_mod(u, u, mz);

  VERIFY(mpz_invert(inv, ctx->M, mz) != 0);

  mpz_mul(u, u, inv);
  mpz_mod(u, u, mz);
  mpz_mul(inv, ctx->M, u);
  mpz_add(ctx->X, ctx->X, inv);
  mpz_mul(ctx->M, ctx->M, mz);

  bool ret = mpz_cmp(ctx->X, Xp) == 0;
  mpz_clears(u, inv, mz, rz, Xp, NULL);
  return ret;
}
