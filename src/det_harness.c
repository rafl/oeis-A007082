#include "oeis.h"
#include "combine.h"
#include "debug.h"
#include "source_combine.h"
#include "source_jack.h"
#include "source_process.h"
#include "maths.h"
#include "primes.h"

#include "interrupt.h"

#include <getopt.h>
#include <gmp.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "gpu_det_real.h"

// #define fld_t uint32_t

// #define FLD_BITS 31

// #define uint128_t __uint128_t


#define POW_CACHE_SPLIT 6
#define POW_CACHE_DIVISOR (1 << POW_CACHE_SPLIT)

static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  int64_t result = k - j;
  return result >= 0 ? (uint64_t)result : result + m;
}


typedef struct {
  uint64_t n, n_args, m,
      m_half; // = (m+1)/2 - used for some jk_sums_pow cache stuff - see
              // jk_sums_pow
  fld_t p,    // n = number of veritcies //obvious
      p_dash, r, r2, r3, // montgomery stuff, a _M suffix implies something is
                         // in montgomery form
      // r is the equivilent of `1` in montgomary space. Which means `r^2` is
      // the equivilent of `r` in montomery space which is mostly used to move
      // numbers into montgomery space (which you do by multplying by r, which
      // is mont_multing by r^2)
      *rs,                  // cache of r^n
      *jk_prod_M,           // cache of w^j*w^-k / (w^-j*w^k + w^j*w^-k)
      *nat_M,               // natural numbers up to n (inclusive)
      *nat_inv_M,           // inverses of natural numbers up to n (inclusive)
      *jk_sums_pow_upper_M, // see jk_sums_pow
      *jk_sums_pow_lower_M,
      *ws_M,       // powers of omega (m form)
      *fact_M,     // i! for i <= n
      *fact_inv_M; // 1/i! for i <= n
} prim_ctx_t;

// shared over threads - not mutated
static prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t n_args, uint64_t m,
                                fld_t p, fld_t w) {
  prim_ctx_t *ctx = malloc(sizeof(prim_ctx_t));
  assert(ctx);
  ctx->n = n;
  ctx->n_args = n_args;
  ctx->m = m;
  ctx->m_half = (ctx->m + 1) / 2;
  ctx->p = p;
  ctx->p_dash = (fld_t)(-inv64_u64(p));
  ctx->r = ((uint128_t)1 << FLD_BITS) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;
  ctx->r3 = (uint128_t)ctx->r2 * ctx->r % p;

  size_t n_rs = (ctx->n_args * ctx->n_args + (FLD_BITS - 1)) / FLD_BITS + 3;
  ctx->rs = malloc(sizeof(fld_t) * n_rs);
  assert(ctx->rs);

  assert((n_args < POW_CACHE_DIVISOR) && "n too big, increase POW_CACHE_SPLIT");

  ctx->rs[0] = 1;
  for (size_t i = 1; i < n_rs; i++) {
    ctx->rs[i] = mont_mul(ctx->rs[i - 1], ctx->r2, ctx->p, ctx->p_dash);
  }

  // initialize roots of unity
  ctx->ws_M = malloc(m * sizeof(fld_t));
  assert(ctx->ws_M);
  ctx->ws_M[0] = ctx->r;
  ctx->ws_M[1] = mont_mul(w, ctx->r2, p, ctx->p_dash);
  for (size_t i = 2; i < m; ++i)
    ctx->ws_M[i] = mont_mul(ctx->ws_M[i - 1], ctx->ws_M[1], p, ctx->p_dash);

  // w^j * w^-k lookup - not actually inserted into the context
  fld_t jk_pairs_M[m * m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_pairs_M[jk_pos(j, k, m)] =
          mont_mul(ctx->ws_M[j], ctx->ws_M[k ? m - k : 0], p, ctx->p_dash);
  }

  // cache of // w^-j*w^k + w^j*w^-k
  fld_t jk_sums_M[m];
  for (size_t k = 0; k < m; ++k) {
    jk_sums_M[jk_pos(0, k, m)] = add_mod_u64(jk_pairs_M[jk_pos(0, k, m)],
                                             jk_pairs_M[jk_pos(k, 0, m)], p);
  }

  // see jk_sums_pow
  ctx->jk_sums_pow_lower_M =
      malloc(sizeof(fld_t) * POW_CACHE_DIVISOR * ctx->m_half);
  assert(ctx->jk_sums_pow_lower_M);
  ctx->jk_sums_pow_upper_M =
      malloc(sizeof(fld_t) * POW_CACHE_DIVISOR * ctx->m_half);
  assert(ctx->jk_sums_pow_upper_M);

  for (size_t j = 0; j < ctx->m_half; j++) {
    ctx->jk_sums_pow_lower_M[j] = ctx->r;
    // we do put w^0 + w^-0 = 2 into this cache currently, but we don't actually
    // use it as we use fast_pow_2 instead.
    ctx->jk_sums_pow_lower_M[ctx->m_half + j] = jk_sums_M[j];
    ctx->jk_sums_pow_upper_M[j] = ctx->r;
    ctx->jk_sums_pow_upper_M[ctx->m_half + j] =
        mont_pow(jk_sums_M[j], POW_CACHE_DIVISOR, ctx->r, ctx->p, ctx->p_dash);
  }

  for (size_t i = 2; i < POW_CACHE_DIVISOR; i++) {
    for (size_t j = 0; j < ctx->m_half; j++) {
      ctx->jk_sums_pow_lower_M[i * ctx->m_half + j] = mont_mul(
          ctx->jk_sums_pow_lower_M[(i - 1) * ctx->m_half + j],
          ctx->jk_sums_pow_lower_M[ctx->m_half + j], ctx->p, ctx->p_dash);
      ctx->jk_sums_pow_upper_M[i * ctx->m_half + j] = mont_mul(
          ctx->jk_sums_pow_upper_M[(i - 1) * ctx->m_half + j],
          ctx->jk_sums_pow_upper_M[ctx->m_half + j], ctx->p, ctx->p_dash);
    }
  }

  // cache of w^j*w^-k / (w^-j*w^k + w^j*w^-k)
  ctx->jk_prod_M = malloc(m * sizeof(fld_t));
  assert(ctx->jk_prod_M);

  for (size_t k = 0; k < m; ++k) {
    size_t pos = jk_pos(0, k, m);
    fld_t sum_inv = mont_inv(jk_sums_M[pos], ctx->r3, p, ctx->p_dash);
    ctx->jk_prod_M[pos] = mont_mul(jk_pairs_M[pos], sum_inv, p, ctx->p_dash);
  }

  // 1 to n
  ctx->nat_M = malloc((n + 1) * sizeof(fld_t));
  assert(ctx->nat_M);
  for (size_t i = 0; i <= n; ++i)
    ctx->nat_M[i] = mont_mul((fld_t)i, ctx->r2, p, ctx->p_dash);

  // 1/i for i = 1 to n
  ctx->nat_inv_M = malloc((n + 1) * sizeof(fld_t));
  assert(ctx->nat_inv_M);
  ctx->nat_inv_M[0] = 0;
  for (size_t k = 1; k <= n; ++k)
    ctx->nat_inv_M[k] = mont_inv(ctx->nat_M[k], ctx->r3, p, ctx->p_dash);

  // i! for i = 1 to n+1
  ctx->fact_M = malloc((n + 1) * sizeof(fld_t));
  assert(ctx->fact_M);
  ctx->fact_M[0] = ctx->r;
  for (size_t i = 1; i < n + 1; ++i)
    ctx->fact_M[i] =
        mont_mul(ctx->fact_M[i - 1], ctx->nat_M[i], p, ctx->p_dash);

  // 1/i! for i = 1 to n+1
  ctx->fact_inv_M = malloc((n + 1) * sizeof(fld_t));
  assert(ctx->fact_inv_M);
  ctx->fact_inv_M[n] = mont_inv(ctx->fact_M[n], ctx->r3, p, ctx->p_dash);
  for (size_t i = n; i; --i)
    ctx->fact_inv_M[i - 1] =
        mont_mul(ctx->fact_inv_M[i], ctx->nat_M[i], p, ctx->p_dash);

  return ctx;
}


static fld_t det_mod_p(fld_t *A, size_t dim, const prim_ctx_t *ctx) {
  const fld_t p = ctx->p, p_dash = ctx->p_dash;
  fld_t det = ctx->r, scaling_factor = ctx->r;

  for (size_t k = 0; k < dim; ++k) {
    size_t pivot_i = k;
    // If the cell on the diagonal we're about to pivot off is zero - find the
    // next row with a non zero entry in that col
    while (pivot_i < dim && A[pivot_i * dim + k] == 0)
      ++pivot_i;
    // if there was no non-zero cell - det is zero

    // This is unreachable except in the JackApprox case
    if (pivot_i == dim)
      return 0;

    // We think this happens almost never
    if (pivot_i != k) {
      // We swap the rows over so that we have a non zero el on the diagonal
      for (size_t j = 0; j < dim; ++j) {
        fld_t tmp = A[k * dim + j];
        A[k * dim + j] = A[pivot_i * dim + j];
        A[pivot_i * dim + j] = tmp;
      }
      det = p - det; // And flip the sign of the determinant
    }

    fld_t pivot = A[k * dim + k];
    // multiply in our value on diagonal
    det = mont_mul(det, pivot, p, p_dash);

    // Now do the elimination
    for (size_t i = k + 1; i < dim; ++i) {
      // Rather than do division on each row we multiply each row up to a common
      // factor scaling factor is where we record the product of thse numbers so
      // we can divide though by at the end to compensate
      scaling_factor = mont_mul(scaling_factor, pivot, p, p_dash);
      fld_t multiplier = A[i * dim + k];
      for (size_t j = k; j < dim; ++j)
        // mul and subtract off the rest
        A[i * dim + j] = mont_mul_sub(A[i * dim + j], pivot, A[k * dim + j],
                                      multiplier, p, p_dash);
    }
  }

  return mont_mul(det, mont_inv(scaling_factor, ctx->r3, p, p_dash), p, p_dash);
}

#define N 39
#define M 21
#define ROWS 20
#define COLS 20
#define NUM_MATRICIES (100 * 1000)
#define P 1073741971

void fill_random_matrices(int *buffer, size_t n, int max_value)
{
    for (size_t m = 0; m < n; ++m) {
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                buffer[m * ROWS * COLS + i * COLS + j] =
                    rand() % (max_value);
            }
        }
    }
}

int main(int argc, char **argv) {
    srand(42);
    

    int * buffer = malloc(sizeof(int32_t) * ROWS * COLS * NUM_MATRICIES);
    int * result_buffer = malloc(sizeof(int32_t) * NUM_MATRICIES);
    fill_random_matrices(buffer, NUM_MATRICIES, P);
    fld_t w = mth_root_mod_p(P, M);

    prim_ctx_t * prim_ctx = prim_ctx_new(N, N, M, P, w);

    // for (int i = 0 ; i <= 32; i++)
    // {
    //     printf("%u\n", det_mod_p(buffer + i * ROWS * COLS, ROWS, prim_ctx));
    // }

    det_mod_p_gpu(buffer, result_buffer, NUM_MATRICIES, prim_ctx->p, prim_ctx->p_dash, prim_ctx->r, prim_ctx->r3);

    for (int i = 0 ; i <= 32; i++)
    {
        printf("%u\n", result_buffer[i]);
    }

  return 0;
}
