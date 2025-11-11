#include "source_process.h"
#include "debug.h"
#include "maths.h"
#include "mss.h"
#include "primes.h"
#include "progress.h"
#include "queue.h"
#include "snapshot.h"

#ifdef USE_GPU
#include "gpu_det.h"
#include <cuda_runtime.h>
#endif

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define POW_CACHE_SPLIT 6
#define POW_CACHE_DIVISOR (1 << POW_CACHE_SPLIT)

static uint64_t m_for(uint64_t n) { return 2 * ((n + 1) / 4) + 1; }

typedef struct {
  uint64_t n, n_args, m,
      m_half, // = (m+1)/2 - used for some jk_sums_pow cache stuff - see
              // jk_sums_pow
      p,      // n = number of veritcies //obvious
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
      *jk_sums_M,           // w^-j*w^k + w^j*w^-k
      *jk_sums_pow_upper_M, // see jk_sums_pow
      *jk_sums_pow_lower_M,
      *ws_M,       // powers of omega (m form)
      *fact_M,     // i! for i <= n
      *fact_inv_M; // 1/i! for i <= n
} prim_ctx_t;

// We have caches of w^-j*w^k + w^j*w^-k and w^j*w^-k / (w^-j*w^k + w^j*w^-k)
// Note that w^-j*w^k = w^(k-j)
// So the value of these expressions is only actually a function of j-k. We use
// this to make the caches smaller and cache more stuff whilst still keeping
// everying in L1
static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  int64_t result = k - j;
  return result >= 0 ? (uint64_t)result : result + m;
}

// shared over threads - not mutated
static prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t n_args, uint64_t m,
                                uint64_t p, uint64_t w) {
  prim_ctx_t *ctx = malloc(sizeof(prim_ctx_t));
  assert(ctx);
  ctx->n = n;
  ctx->n_args = n_args;
  ctx->m = m;
  ctx->m_half = (ctx->m + 1) / 2;
  ctx->p = p;
  ctx->p_dash = (uint64_t)(-inv64_u64(p));
  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;
  ctx->r3 = (uint128_t)ctx->r2 * ctx->r % p;

  size_t n_rs = (ctx->n_args * ctx->n_args + 63) / 64 + 3;
  ctx->rs = malloc(sizeof(uint64_t) * n_rs);
  assert(ctx->rs);

  assert((n_args < POW_CACHE_DIVISOR) && "n too big, increase POW_CACHE_SPLIT");

  ctx->rs[0] = 1;
  for (size_t i = 1; i < n_rs; i++) {
    ctx->rs[i] = mont_mul(ctx->rs[i - 1], ctx->r2, ctx->p, ctx->p_dash);
  }

  // initialize roots of unity
  ctx->ws_M = malloc(m * sizeof(uint64_t));
  assert(ctx->ws_M);
  ctx->ws_M[0] = ctx->r;
  ctx->ws_M[1] = mont_mul(w, ctx->r2, p, ctx->p_dash);
  for (size_t i = 2; i < m; ++i)
    ctx->ws_M[i] = mont_mul(ctx->ws_M[i - 1], ctx->ws_M[1], p, ctx->p_dash);

  // w^j * w^-k lookup - not actually inserted into the context
  uint64_t jk_pairs_M[m * m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_pairs_M[jk_pos(j, k, m)] =
          mont_mul(ctx->ws_M[j], ctx->ws_M[k ? m - k : 0], p, ctx->p_dash);
  }

  // cache of // w^-j*w^k + w^j*w^-k
  ctx->jk_sums_M = malloc(m * sizeof(uint64_t));
  assert(ctx->jk_sums_M);
  for (size_t k = 0; k < m; ++k) {
    ctx->jk_sums_M[jk_pos(0, k, m)] = add_mod_u64(
        jk_pairs_M[jk_pos(0, k, m)], jk_pairs_M[jk_pos(k, 0, m)], p);
  }

  // see jk_sums_pow
  ctx->jk_sums_pow_lower_M =
      malloc(sizeof(uint64_t) * POW_CACHE_DIVISOR * ctx->m_half);
  assert(ctx->jk_sums_pow_lower_M);
  ctx->jk_sums_pow_upper_M =
      malloc(sizeof(uint64_t) * POW_CACHE_DIVISOR * ctx->m_half);
  assert(ctx->jk_sums_pow_upper_M);

  for (size_t j = 0; j < ctx->m_half; j++) {
    ctx->jk_sums_pow_lower_M[j] = ctx->r;
    // we do put w^0 + w^-0 = 2 into this cache currently, but we don't actually
    // use it as we use fast_pow_2 instead.
    ctx->jk_sums_pow_lower_M[ctx->m_half + j] = ctx->jk_sums_M[j];
    ctx->jk_sums_pow_upper_M[j] = ctx->r;
    ctx->jk_sums_pow_upper_M[ctx->m_half + j] = mont_pow(
        ctx->jk_sums_M[j], POW_CACHE_DIVISOR, ctx->r, ctx->p, ctx->p_dash);
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
  ctx->jk_prod_M = malloc(m * sizeof(uint64_t));
  assert(ctx->jk_prod_M);

  for (size_t k = 0; k < m; ++k) {
    size_t pos = jk_pos(0, k, m);
    uint64_t sum_inv = mont_inv(ctx->jk_sums_M[pos], ctx->r3, p, ctx->p_dash);
    ctx->jk_prod_M[pos] = mont_mul(jk_pairs_M[pos], sum_inv, p, ctx->p_dash);
  }

  // 1 to n
  ctx->nat_M = malloc((n + 1) * sizeof(uint64_t));
  assert(ctx->nat_M);
  for (size_t i = 0; i <= n; ++i)
    ctx->nat_M[i] = mont_mul(i, ctx->r2, p, ctx->p_dash);

  // 1/i for i = 1 to n
  ctx->nat_inv_M = malloc((n + 1) * sizeof(uint64_t));
  assert(ctx->nat_inv_M);
  ctx->nat_inv_M[0] = 0;
  for (size_t k = 1; k <= n; ++k)
    ctx->nat_inv_M[k] = mont_inv(ctx->nat_M[k], ctx->r3, p, ctx->p_dash);

  // i! for i = 1 to n+1
  ctx->fact_M = malloc((n + 1) * sizeof(uint64_t));
  assert(ctx->fact_M);
  ctx->fact_M[0] = ctx->r;
  for (size_t i = 1; i < n + 1; ++i)
    ctx->fact_M[i] =
        mont_mul(ctx->fact_M[i - 1], ctx->nat_M[i], p, ctx->p_dash);

  // 1/i! for i = 1 to n+1
  ctx->fact_inv_M = malloc((n + 1) * sizeof(uint64_t));
  assert(ctx->fact_inv_M);
  ctx->fact_inv_M[n] = mont_inv(ctx->fact_M[n], ctx->r3, p, ctx->p_dash);
  for (size_t i = n; i; --i)
    ctx->fact_inv_M[i - 1] =
        mont_mul(ctx->fact_inv_M[i], ctx->nat_M[i], p, ctx->p_dash);

  return ctx;
}

static void prim_ctx_free(prim_ctx_t *ctx) {
  free(ctx->rs);
  free(ctx->fact_inv_M);
  free(ctx->fact_M);
  free(ctx->ws_M);
  free(ctx->jk_sums_M);
  free(ctx->nat_inv_M);
  free(ctx->nat_M);
  free(ctx->jk_prod_M);
  free(ctx->jk_sums_pow_lower_M);
  free(ctx->jk_sums_pow_upper_M);
  free(ctx);
}

// Raising coputing power of 2 is easy. Just if we'd overflow 2^64
// we need to modular divide down. 2^64 = r though. So we use a
// cache for r^n for how many times we overflow 2^64, and then a
// left shift for the rest.
uint64_t fast_pow_2(const prim_ctx_t *ctx, uint64_t pow) {
  uint64_t r_pow = pow / 64;
  uint64_t remain = pow % 64;
  uint64_t pow2 = 1UL << remain;

  return mont_mul(pow2, ctx->rs[r_pow + 2], ctx->p, ctx->p_dash);
}

// Note that w^i + w^-i = w^-i + w^i
// So there's only actually (m+1)/2 unique values for w^-i + w^i which lets us
// shrink our cache size in half. The "diff" value is the index into this cache
//
// Raising w^-j*w^k + w^j*w^-k to intger powers is a thing we do often
// so it's nice to just cache the results. The powers we need to go up to are
// n_args * n_args. n_args * n_args * (m+1/2) * sizeof(uint64_t)
// is getting kinda large for large values of n_args. I don't want the code to
// only be efficient on machines with large L1 cache. So split the cache into
// two parts. A cache of jk_sums^i for i < POW_CACHE_DIVISOR and a cache of
// (jk_sums^POW_CACHE_DIVISOR)^i We can then do two cache lookups and a single
// multiply
int64_t jk_sums_pow(const prim_ctx_t *ctx, uint64_t diff, uint64_t pow) {
  uint64_t upper_index = pow >> POW_CACHE_SPLIT;
  uint64_t lower_index = pow & (POW_CACHE_DIVISOR - 1);

  uint64_t upper_index_full = upper_index * ctx->m_half + diff;
  uint64_t lower_index_full = lower_index * ctx->m_half + diff;

  return mont_mul(ctx->jk_sums_pow_upper_M[upper_index_full],
                  ctx->jk_sums_pow_lower_M[lower_index_full], ctx->p,
                  ctx->p_dash);
}

// Calculate the multinomal coefficient where the powers of x_i are given by *ms
static uint64_t multinomial_mod_p(const prim_ctx_t *ctx, const size_t *ms,
                                  size_t len) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;

  uint64_t coeff = ctx->fact_M[ctx->n_args - 1];
  for (size_t i = 0; i < len; ++i)
    coeff = mont_mul(coeff, ctx->fact_inv_M[ms[i]], p, p_dash);

  return coeff;
}

// Compute the determinant via gaussian elimination
// we cary the denominator through and only compute a single inverse at the end
static uint64_t det_mod_p(uint64_t *A, size_t dim, const prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;
  uint64_t det = ctx->r, scaling_factor = ctx->r;

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
        uint64_t tmp = A[k * dim + j];
        A[k * dim + j] = A[pivot_i * dim + j];
        A[pivot_i * dim + j] = tmp;
      }
      det = p - det; // And flip the sign of the determinant
    }

    uint64_t pivot = A[k * dim + k];
    // multiply in our value on diagonal
    det = mont_mul(det, pivot, p, p_dash);

    // Now do the elimination
    for (size_t i = k + 1; i < dim; ++i) {
      // Rather than do division on each row we multiply each row up to a common
      // factor scaling factor is where we record the product of thse numbers so
      // we can divide though by at the end to compensate
      scaling_factor = mont_mul(scaling_factor, pivot, p, p_dash);
      uint64_t multiplier = A[i * dim + k];
      for (size_t j = k; j < dim; ++j)
        // mul and subtract off the rest
        A[i * dim + j] = mont_mul_sub(A[i * dim + j], pivot, A[k * dim + j],
                                      multiplier, p, p_dash);
    }
  }

  return mont_mul(det, mont_inv(scaling_factor, ctx->r3, p, p_dash), p, p_dash);
}

static uint64_t jack_snd_trm(uint64_t *c, const prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, m = ctx->m;

  // active groups
  // This is the indexs of the non zero elements of c
  // Which is also the powers of w they correspond to
  // typ for "type" aka class / colour
  size_t typ[m], r = 0;
  for (size_t i = 0; i < m; ++i) {
    if (c[i]) {
      typ[r] = i;
      ++r;
    }
  }

  uint64_t prod_M = ctx->r;

  // for each non zero power of omega w^i in our args
  for (size_t a = 0; a < r; ++a) {
    size_t i = typ[a];

    // Jack wants to add 1 to the diagonal
    uint64_t sum = ctx->r;
    // for each non zero power of omega w^j in our args
    for (size_t b = 0; b < r; ++b) {
      size_t j = typ[b];

      // look up w^j*w^-k / (w^-j*w^k + w^j*w^-k)
      uint64_t w = ctx->jk_prod_M[jk_pos(i, j, m)];

      // sum is gunna be our diagonal element for this row / the row sum
      sum = add_mod_u64(sum, mont_mul(ctx->nat_M[c[j]], w, p, ctx->p_dash), p);
    }

    // so prod m is the product of the row sums of the rows we're deleting
    prod_M = mont_pow(sum, c[i] - 1, prod_M, p, ctx->p_dash);
  }

  // we divide prod_M by the multiplicty of 1 because... ?
  // Probably something to do with us not dropping one of the ones or sth?
  // I'll try taking it out
  // prod_M = mont_mul(prod_M, ctx->nat_inv_M[c[0]], p, ctx->p_dash);

  // If all terms were the same power of w and we quotiented everything out
  if (r <= 1)
    return prod_M;

  // We're constructing the minor of the reduced matrix
  uint64_t A[r * r];
  for (size_t a = 0; a < r; ++a) {
    // looking t the w^i args
    size_t i = typ[a];

    // look up w^j*w^-k / (w^-j*w^k + w^j*w^-k) for i, 0...

    // uint64_t W_del = ctx->jk_prod_M[jk_pos(i, 0, m)];

    // we're taking the multiplicity of 1 * this term
    // Jack wants this to start at 1 again
    uint64_t diag = ctx->r; // mont_mul(ctx->nat_M[c[0]], W_del, p,
                            // ctx->p_dash);

    // remaining off-diagonal blocks
    for (size_t b = 0; b < r; ++b) {
      size_t j = typ[b];
      // going over w^j args

      if (j == i)
        continue;

      // we're we're treading the 1 term as special... yeah ok I guess it's
      // because it doesn't go in the matrix? so this is the new "w del"
      uint64_t w = ctx->jk_prod_M[jk_pos(i, j, m)];

      // Again fill the matrix as per it's normal terms but with multiplicity
      uint64_t v = mont_mul(ctx->nat_M[c[j]], w, p, ctx->p_dash);

      // This is the -1 coefficient on the off diag elements
      A[(a)*r + (b)] = p - v;
      // and add it to the total for the diag elements
      diag = add_mod_u64(diag, v, p);
    }

    A[(a)*r + (a)] = diag;
  }

  // I guess prod_M is some magic compensation coefficient
  uint64_t ret = mont_mul(prod_M, det_mod_p(A, r, ctx), p, ctx->p_dash);
  return ret;
}

// Multiplying through all the x_i*x_j^-1 + x_i^-1*x_j terms
static uint64_t f_fst_trm(uint64_t *c, const prim_ctx_t *ctx) {
  const uint64_t m = ctx->m, p = ctx->p, p_dash = ctx->p_dash;

  uint64_t e = 0;
  // Now we go throug the cases where x_i != x_j
  // We note that w^j w^-k = w^(j-k)
  // and also that w^i + w^-i = w^-i + w^i
  // so we keep track how often each of the (m+1)/2 possible terms happens in
  // pows and then compute the power at the end
  uint64_t pows[ctx->m];
  memset(pows, 0, sizeof(uint64_t) * ctx->m);

  for (size_t a = 0; a < m; ++a) {
    uint64_t ca = c[a];
    if (!ca)
      continue;

    // This is going through all the cases where x_i == x_j
    // We'll just count how often this happens and then raise
    // w^0 + w^-0 = 2 to that power
    e += (ca * (ca - 1));

    for (size_t b = a + 1; b < m; ++b) {
      uint64_t cb = c[b];

      // Maybe a slightly better way to do this
      uint64_t diff = b - a;
      pows[diff] += ca * cb;
    }
  }

  uint64_t acc = fast_pow_2(ctx, e / 2);

  // actually the (w^0 + w^-0 term) was already accounted for earier
  // so the i=0 term is always zero and we can skip it
  for (size_t i = 1; i < ctx->m_half; i++) {
    uint64_t pow_val = jk_sums_pow(ctx, i, pows[i] + pows[m - i]);

    acc = mont_mul(acc, pow_val, p, p_dash);
  }

  return acc;
}

static uint64_t jack_offset(uint64_t *vec, const prim_ctx_t *ctx) {
  return mont_mul(f_fst_trm(vec, ctx), jack_snd_trm(vec, ctx), ctx->p,
                  ctx->p_dash);
}

static uint64_t jack(uint64_t *vec, const prim_ctx_t *ctx) {
  uint64_t const m = ctx->m, p = ctx->p;
  uint64_t ret = 0;
  uint64_t f_0 = jack_offset(vec, ctx);
  uint64_t const coeff_baseline = multinomial_mod_p(ctx, vec, m);

  // Loop over each "rotation" of the vector of argument multiplicities. This is
  // equivilent to multiplying all the coefficients by w
  for (size_t r = 0; r < m; ++r) {
    // We require there always be at least one "1" in the arguments to f() (per
    // the paper) that is to say if the multiplicty of "1" arguments is zero -
    // we should skip this case
    if (vec[r] == 0)
      continue;

    // The multinomial coefficient would be constant over all "rotations" of the
    // multiplicities but because we're assuming at least one argument is always
    // "1" which requires us to subtract 1 from the first multiplicity. Rather
    // than recompute the full coeff each time we can take a baseline
    // "coefficient" and multiply it by j to convert 1/j! to 1/(j-1!)
    size_t coeff = mont_mul(coeff_baseline, ctx->nat_M[vec[r]], p, ctx->p_dash);
    uint64_t f_n = mont_mul(coeff, f_0, p, ctx->p_dash);
    ret = add_mod_u64(ret, f_n, p);
  }

  ret = mont_mul(ret, ctx->nat_M[ctx->n - 1], ctx->p, ctx->p_dash);

  // Second multiply is due to not computing the extra col in det:
  return mont_mul(ret, ctx->nat_M[ctx->n - 1], ctx->p, ctx->p_dash);
}

static uint64_t f_snd_trm(uint64_t *c, const prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, m = ctx->m;

  // active groups typ for "type" aka class / colour
  // This is the indexes of the non zero elements of c
  // Which is also the powers of w they correspond to
  size_t typ[m], r = 0;
  for (size_t i = 0; i < m; ++i) {
    if (c[i]) {
      typ[r] = i;
      ++r;
    }
  }

  uint64_t prod_M = ctx->r;

  // for each i where w^i has non zero multiplicity in our args
  for (size_t a = 0; a < r; ++a) {
    size_t i = typ[a];
    if (c[i] == 1) {
      continue;
    }

    // This is similar to row sum of the off diagonal term in the full matrix
    // The off diagonal term would have one subtracted from multiplicity when j
    // = i (as there's no edge E_ii)
    uint64_t sum = 0;

    // for each j where w^j has non zero multiplicity in our args
    for (size_t b = 0; b < r; ++b) {
      size_t j = typ[b];

      // This is the inner element of A matrix from the paper (up to sign) /
      // inner element of product sum in paper
      uint64_t W = ctx->jk_prod_M[jk_pos(i, j, m)];

      // sum += W * multiplicity of (w^j)

      // Try cache lookup?
      sum = add_mod_u64(sum, mont_mul(ctx->nat_M[c[j]], W, p, ctx->p_dash), p);
    }

    // prod_M then is the multiple of all these row sums to the power of
    // multiplicity-1 i.e. the product for the row sum for each deleted row
    prod_M = mont_pow(sum, c[i] - 1, prod_M, p, ctx->p_dash);
  }

  // Now we divide prod_M by the multiplicity of 1 because???
  // I assume it has to do with the fact that's the col dropped in our minor
  prod_M = mont_mul(prod_M, ctx->nat_inv_M[c[0]], p, ctx->p_dash);

  // the dimension (width/height) of the minor is 1 less than dim of quotient
  // matrix
  size_t dim = r - 1;

  // If all terms were the same power of w and we quotiented everything out
  if (!dim)
    return prod_M;

  // We're constructing the minor of the quotiented matrix
  // first row / col is the one being dropped the one corresponding to the w^0 =
  // 1 argument

  uint64_t A[dim * dim];
  // for each i!=0 where w^i has non zero multiplicity in our args
  for (size_t a = 1; a < r; ++a) {
    size_t i = typ[a];

    // contribution from the column removed to form the minor (that's a consant
    // 1 - so w^0 )
    uint64_t W_del = ctx->jk_prod_M[m - i];

    // When making the "laplaican" the diagonal is the sum of all off-diagonal
    // elements in the row of the full matrix including the column dropped to
    // form the minor. So we add that to the diag here (with multiplicity)
    uint64_t diag = mont_mul(ctx->nat_M[c[0]], W_del, p, ctx->p_dash);

    // for each j !=0 where w^j has non zero multiplicity in our args
    for (size_t b = 1; b < r; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      uint64_t W = ctx->jk_prod_M[jk_pos(i, j, m)];

      // Could do a lookup here
      uint64_t v = mont_mul(ctx->nat_M[c[j]], W, p, ctx->p_dash);

      // This is the -1 coefficient on the off diag elements

      // We could flip the sign here to avoid some subtractions...
      A[(a - 1) * dim + (b - 1)] = p - v;
      // and add it to the total for the diag elements
      diag = add_mod_u64(diag, v, p);
    }

    A[(a - 1) * dim + (a - 1)] = diag;
  }

  // I guess prod_M is some magic compensation coefficient
  return mont_mul(prod_M, det_mod_p(A, dim, ctx), p, ctx->p_dash);
}

static uint64_t f(uint64_t *vec, const prim_ctx_t *ctx) {
  return mont_mul(f_fst_trm(vec, ctx), f_snd_trm(vec, ctx), ctx->p,
                  ctx->p_dash);
}

static uint64_t david(uint64_t *vec, const prim_ctx_t *ctx) {
  uint64_t const m = ctx->m, p = ctx->p;
  uint64_t ret = 0, f_0 = f(vec, ctx);
  uint64_t const coeff_baseline = multinomial_mod_p(ctx, vec, m);

  // Loop over each "rotation" of the vector of argument multiplicities. This is
  // equivilent to multiplying all the coefficients by w
  for (size_t r = 0; r < m; ++r) {
    // We require there always be at least one "1" in the arguments to f() (per
    // the paper) that is to say if the multiplicty of "1" arguments is zero -
    // we should skip this case
    if (vec[r] == 0)
      continue;

    // The multinomial coefficient would be constant over all "rotations" of the
    // multiplicities but because we're assuming at least one argument is always
    // "1" which requires us to subtract 1 from the first multiplicity. Rather
    // than recompute the full coeff each time we can take a baseline
    // "coefficient" and multiply it by j to convert 1/j! to 1/(j-1!)
    size_t coeff = mont_mul(coeff_baseline, ctx->nat_M[vec[r]], p, ctx->p_dash);

    size_t idx = (2 * r) % m;
    // f_0 = coeff * f_0 * w^(m-idx) = coeff * f_0 * w^-2r
    // This result comes from having to permute one of the ones from one of the
    // first n-1 args into the nth arg See the paper for more info
    uint64_t f_n = mont_mul(
        coeff, mont_mul(f_0, ctx->ws_M[idx ? m - idx : 0], p, ctx->p_dash), p,
        ctx->p_dash);

    ret = add_mod_u64(ret, f_n, p);
  }

  return ret;
}


// state of whole process (shared over threads)
typedef struct {
  process_mode_t mode;
  uint64_t n;      /* element of seq */
  uint64_t n_args; // Number of arguments `f` or (or `f_jack_offset`) takes
  uint64_t m;      /*w is the mth root of unity*/
  uint64_t *ps;    /* list of primes */
  size_t idx /* withth prime*/, np /* number of primes*/,
      *vecss /*Some buffers for vectors*/;
  bool quiet, snapshot,
      borrowed;   /*mode stuff - saving snapshots so you can restart*/
  size_t n_thrds; /*number of threads*/
} proc_state_t;

typedef struct {
  uint64_t (*f)(uint64_t *, const prim_ctx_t *);
  _Atomic size_t *done;
  const prim_ctx_t *ctx;
  queue_t *q;
  size_t *vecs;
  uint64_t *l_acc, *acc;
  pthread_mutex_t *acc_mu;
  bool idle;
  int device_id;  // GPU device ID for this worker
  size_t prefix_depth;
  size_t worker_id;  // Unique worker index (for batch allocation)
#ifdef USE_GPU
  gpu_shared_ctx_t *gpu_shared;  // Shared GPU context (NULL for CPU workers)
#endif
} worker_t;

static void w_resume(void *ud) {
  worker_t *w = ud;
  w->idle = false;
}

static resume_cb_t w_idle(void *ud) {
  worker_t *w = ud;
  pthread_mutex_lock(w->acc_mu);
  *w->acc = add_mod_u64(*w->acc, *w->l_acc, w->ctx->p);
  *w->l_acc = 0;
  pthread_mutex_unlock(w->acc_mu);
  w->idle = true;
  return w_resume;
}

// thread function for doing work on the "maths stuff" - CPU version with local iterators
static void *residue_for_prime(void *ud) {
  worker_t *worker = ud;
  uint64_t (*f)(uint64_t *, const prim_ctx_t *) = worker->f;
  const prim_ctx_t *ctx = worker->ctx;
  uint64_t m = ctx->m, p = ctx->p,
           l_acc = 0; // l_acc is where we're going to accumulate the total
                      // residual from the stuff we pull from our work queue
  size_t *prefix_buf = worker->vecs;
  size_t prefix_depth = worker->prefix_depth;
  size_t n_args = ctx->n_args;
  worker->l_acc = &l_acc;

  size_t sub_scratch[m + 1], full_vec[m], necklace_count = 0;

  for (;;) {
    size_t n_vec = queue_pop(worker->q, prefix_buf, w_idle, worker);
    if (!n_vec)
      break;

    for (size_t c = 0; c < n_vec; ++c) {
      size_t *prefix = &prefix_buf[c * prefix_depth];

      // Create local iterator from this prefix
      canon_iter_t sub_iter;
      canon_iter_from_prefix(&sub_iter, m, n_args, sub_scratch, prefix,
                             prefix_depth);

      // Iterate through all full vectors for this prefix
      while (canon_iter_next(&sub_iter, full_vec)) {
        // each vector has len m
        // each vector contains the multiplicity with which each power of omega
        // appears in the args i.e. 1 3 7 = 1 lot of w^0, 3 lots of w^1, 7
        // lots of w^2
        l_acc = add_mod_u64(l_acc, f(full_vec, ctx), p);
        ++necklace_count;
      }
    }
    atomic_fetch_add_explicit(worker->done, necklace_count, memory_order_relaxed);
    necklace_count = 0;
  }

  (void)w_idle(worker);
  return NULL;
}

#ifdef USE_GPU
// GPU batch size - independent of queue CHUNK size
// Controls how many full vectors are sent to GPU at once
#define GPU_BATCH_SIZE (1UL << 12)

// Helper to post-process a batch of results
static void process_batch_results(worker_t *worker, vec_batch_t *batch,
                                   size_t n_vec, uint64_t *l_acc) {
  const prim_ctx_t *ctx = worker->ctx;
  uint64_t p = ctx->p;

  // GPU has computed complete david() or jack() result for each vector
  // Just retrieve and accumulate
  for (size_t c = 0; c < n_vec; ++c) {
    uint64_t result = vec_batch_get(batch, c);
    *l_acc = add_mod_u64(*l_acc, result, p);
  }

  atomic_fetch_add_explicit(worker->done, n_vec, memory_order_relaxed);
}

// GPU-accelerated worker with prefix-based batching
static void *residue_for_prime_gpu(void *ud) {
  worker_t *worker = ud;
  const prim_ctx_t *ctx = worker->ctx;
  uint64_t m = ctx->m, l_acc = 0;
  size_t n_args = ctx->n_args;
  size_t prefix_depth = worker->prefix_depth;
  worker->l_acc = &l_acc;

  // Set GPU device for this worker
  cudaSetDevice(worker->device_id);

  // Use blocking sync instead of busy-wait to avoid wasting CPU cycles
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  // Create 4 GPU batches for quad buffering using shared context
  // More buffers = more in-flight work = better GPU utilization
  #define NUM_BUFFERS 4
  vec_batch_t *batches[NUM_BUFFERS];
  size_t *full_vecs_buffers[NUM_BUFFERS];
  size_t n_vecs[NUM_BUFFERS];

  // Each worker gets NUM_BUFFERS consecutive batch indices from the shared pool
  size_t batch_start = worker->worker_id * NUM_BUFFERS;
  for (int i = 0; i < NUM_BUFFERS; i++) {
    batches[i] = vec_batch_new(worker->gpu_shared, batch_start + i);
    full_vecs_buffers[i] = malloc(GPU_BATCH_SIZE * m * sizeof(size_t));
    n_vecs[i] = 0;
  }

  // Allocate buffers for prefix expansion
  size_t *prefix_buf = worker->vecs;
  size_t sub_scratch[m + 1];
  size_t full_vec[m];

  int current = 0;
  int in_flight = 0;  // How many batches are currently in flight on GPU
  bool queue_done = false;
  size_t n_prefixes = 0, prefix_idx = 0;
  bool iter_active = false;  // Track if we have a partial iterator
  canon_iter_t sub_iter;     // Persistent iterator across batches

  // Helper macro to fill a GPU batch from prefixes
  #define FILL_BATCH(buf_idx, result_var) do { \
    vec_batch_clear(batches[buf_idx]); \
    n_vecs[buf_idx] = 0; \
    \
    while (n_vecs[buf_idx] < GPU_BATCH_SIZE && !queue_done) { \
      if (!iter_active) { \
        if (prefix_idx >= n_prefixes) { \
          n_prefixes = queue_pop(worker->q, prefix_buf, w_idle, worker); \
          prefix_idx = 0; \
          if (n_prefixes == 0) { \
            queue_done = true; \
            break; \
          } \
        } \
        \
        size_t *prefix = &prefix_buf[prefix_idx * prefix_depth]; \
        canon_iter_from_prefix(&sub_iter, m, n_args, sub_scratch, prefix, prefix_depth); \
        iter_active = true; \
      } \
      \
      if (canon_iter_next(&sub_iter, full_vec)) { \
        memcpy(&full_vecs_buffers[buf_idx][n_vecs[buf_idx] * m], full_vec, m * sizeof(size_t)); \
        vec_batch_add(batches[buf_idx], &full_vecs_buffers[buf_idx][n_vecs[buf_idx] * m]); \
        n_vecs[buf_idx]++; \
      } else { \
        iter_active = false; \
        prefix_idx++; \
      } \
    } \
    \
    result_var = (n_vecs[buf_idx] > 0); \
  } while(0)

  for (;;) {
    // Fill up the pipeline initially
    while (in_flight < NUM_BUFFERS) {
      bool has_work;
      FILL_BATCH(current, has_work);
      if (!has_work) {
        // No more work
        break;
      }

      // Launch async GPU compute on current batch (returns immediately)
      vec_batch_compute_async(batches[current]);
      in_flight++;
      current = (current + 1) % NUM_BUFFERS;
    }

    if (in_flight == 0) {
      // Never got any work
      break;
    }

    // Now process the oldest batch (which should be done or nearly done)
    int oldest = (current - in_flight + NUM_BUFFERS) % NUM_BUFFERS;

    // Wait for oldest batch to complete
    vec_batch_wait(batches[oldest]);

    // Process its results
    process_batch_results(worker, batches[oldest], n_vecs[oldest], &l_acc);
    in_flight--;

    // Try to fill and launch a new batch
    bool has_work;
    FILL_BATCH(current, has_work);
    if (!has_work) {
      // No more work, drain remaining in-flight batches
      while (in_flight > 0) {
        oldest = (current - in_flight + NUM_BUFFERS) % NUM_BUFFERS;
        vec_batch_wait(batches[oldest]);
        process_batch_results(worker, batches[oldest], n_vecs[oldest], &l_acc);
        in_flight--;
      }
      break;
    }

    // Launch async GPU compute on current batch
    vec_batch_compute_async(batches[current]);
    in_flight++;
    current = (current + 1) % NUM_BUFFERS;
  }

  for (int i = 0; i < NUM_BUFFERS; i++) {
    vec_batch_free(batches[i]);
    free(full_vecs_buffers[i]);
  }
  #undef FILL_BATCH
  #undef NUM_BUFFERS

  (void)w_idle(worker);
  return NULL;
}
#endif

static size_t get_num_threads() {
  char *env = getenv("OMP_NUM_THREADS");
  if (env) {
    char *endptr;
    long val = strtol(env, &endptr, 10);
    if (*endptr == '\0' && val > 0)
      return (size_t)val;
  }

  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (size_t)n : 1;
}

static int ret(proc_state_t *st, prim_ctx_t *ctx, uint64_t acc, uint64_t *res,
               uint64_t *p_ret) {
  uint64_t denom = mont_inv(mont_pow(ctx->nat_M[ctx->m], ctx->n_args - 1,
                                     ctx->r, ctx->p, ctx->p_dash),
                            ctx->r3, ctx->p, ctx->p_dash);
  uint64_t ret = mont_mul(mont_mul(acc, denom, ctx->p, ctx->p_dash), 1, ctx->p,
                          ctx->p_dash);
  prim_ctx_free(ctx);

  *p_ret = st->ps[st->idx++];
  *res = ret;
  return 1;
}

// Implementation of the virtual function - entrypoint for calculation
static int proc_next(source_t *self, uint64_t *res, uint64_t *p_ret) {
  proc_state_t *st = self->state;
  if (st->idx == st->np)
    return 0;

  // n, m, prime
  uint64_t n = st->n, m = st->m, p = st->ps[st->idx];
  // root of unity
  uint64_t w = mth_root_mod_p(p, m);
  prim_ctx_t *ctx = prim_ctx_new(n, st->n_args, m, p, w);

  const size_t siz = canon_iter_size(m, st->n_args);

  _Atomic size_t done = 0;
  uint64_t acc = 0;

  uint64_t iter_st[m + 5];
  size_t st_len = 0;

  if (st->snapshot)
    snapshot_try_resume(st->mode, n, p, &done, &acc, iter_st, &st_len);
  assert(done <= siz);

  if (done == siz)
    return ret(st, ctx, acc, res, p_ret);

  size_t prefix_depth = canon_iter_depth_for(m);

  // shared work queue
  queue_t *q = queue_new(st->n_args, m, prefix_depth, iter_st, st_len,
                         &st->vecss[st->n_thrds * CHUNK * m]);

  progress_t prog;
  // progress bar stuff
  if (!st->quiet)
    progress_start(&prog, p, &done, siz, &q->fill);

  // make some threads
  pthread_mutex_t acc_mu = PTHREAD_MUTEX_INITIALIZER;
  uint64_t (*fn)(uint64_t *, const prim_ctx_t *) =
      st->mode == PROC_MODE_JACK_OFFSET ? jack : david;

#ifdef USE_GPU
  bool use_gpu = gpu_det_available();
  int n_gpus = use_gpu ? gpu_det_device_count() : 0;

  // Create shared GPU contexts (one per GPU) with constant lookup tables and buffer pools
  gpu_shared_ctx_t *gpu_shared_ctxs[n_gpus];
  if (use_gpu) {
    bool is_jack_mode = (fn == jack);
    size_t n_rs = (ctx->n_args * ctx->n_args + 63) / 64 + 3;
    // Allocate enough batches for all threads (worst case all on one GPU)
    // Each worker needs NUM_BUFFERS (4) batches
    size_t max_batches = st->n_thrds * 4;
    for (int gpu_id = 0; gpu_id < n_gpus; ++gpu_id) {
      gpu_shared_ctxs[gpu_id] = gpu_shared_ctx_new(
          gpu_id, ctx->n, ctx->n_args, ctx->m, ctx->p, ctx->p_dash,
          ctx->r, ctx->r3, ctx->jk_prod_M, ctx->nat_M, ctx->nat_inv_M,
          ctx->ws_M, ctx->jk_sums_M, ctx->jk_sums_pow_lower_M,
          ctx->jk_sums_pow_upper_M, ctx->rs, ctx->fact_M, ctx->fact_inv_M,
          ctx->m_half, n_rs, is_jack_mode, max_batches, GPU_BATCH_SIZE);
    }
  }

  // With GPU: use fewer workers (2-3 per GPU) to reduce queue contention
  // Remaining threads will be producers
  size_t n_workers = use_gpu ? ((size_t)n_gpus * 4) : st->n_thrds;
  if (n_workers > st->n_thrds) n_workers = st->n_thrds;
  void *(*worker_fn)(void *) = use_gpu ? residue_for_prime_gpu : residue_for_prime;
#else
  size_t n_workers = st->n_thrds;
  void *(*worker_fn)(void *) = residue_for_prime;
#endif

  pthread_t worker[n_workers];
  worker_t w_ctxs[n_workers];
  bool *idles[n_workers];

  for (size_t i = 0; i < n_workers; ++i) {
    w_ctxs[i] = (worker_t){.ctx = ctx,
                           .f = fn,
                           .done = &done,
                           .q = q,
                           .vecs = &st->vecss[i * CHUNK * m],
                           .idle = false,
                           .acc = &acc,
                           .acc_mu = &acc_mu,
                           .prefix_depth = prefix_depth,
                           .worker_id = i,
#ifdef USE_GPU
                           .device_id = use_gpu ? (int)(i % n_gpus) : 0,
                           .gpu_shared = use_gpu ? gpu_shared_ctxs[i % n_gpus] : NULL
#else
                           .device_id = 0
#endif
                           };
    idles[i] = &w_ctxs[i].idle;
    // worker threads
    pthread_create(&worker[i], NULL, worker_fn, &w_ctxs[i]);
  }

  snapshot_t ss;
  if (st->snapshot)
    snapshot_start(&ss, st->mode, n, p, n_workers, q, idles, &done, &acc);

  queue_fill(q);

  for (size_t i = 0; i < n_workers; ++i)
    pthread_join(worker[i], NULL);

#ifdef USE_GPU
  // Free shared GPU contexts
  if (use_gpu) {
    for (int gpu_id = 0; gpu_id < n_gpus; ++gpu_id) {
      gpu_shared_ctx_free(gpu_shared_ctxs[gpu_id]);
    }
  }
#endif

  queue_free(q);
  if (st->snapshot)
    snapshot_stop(&ss);

  if (!st->quiet)
    progress_stop(&prog);

  return ret(st, ctx, acc, res, p_ret);
}

size_t *source_process_vecss(source_t *self) {
  proc_state_t *st = self->state;
  return st->vecss;
}

static void proc_destroy(source_t *self) {
  proc_state_t *st = self->state;
  if (!st->borrowed)
    free(st->vecss);
  free(st->ps);
  free(st);
  free(self);
}

#define P_STRIDE (1ULL << 10)

source_t *source_process_new(process_mode_t mode, uint64_t n, uint64_t m_id,
                             bool quiet, bool snapshot, size_t *vecss) {
  uint64_t m = m_for(n);
  assert(mode <= PROC_MODE_JACKEST);
  if (mode == PROC_MODE_JACKEST || mode == PROC_MODE_JACK_OFFSET) {
    assert("jack modes are only valid when n%4 == 3 && n > 3" &&
           (n > 3 && n % 4 == 3));
    m -= 2;
  }

  uint64_t n_args = (mode == PROC_MODE_JACK_OFFSET) ? n - 2 : n;
  size_t np;
  assert(m_id < P_STRIDE);
  uint64_t *ps = build_prime_list(n, m, m_id, P_STRIDE, &np);

  proc_state_t *st = malloc(sizeof(*st));
  assert(st);
  size_t n_thrds = get_num_threads();
  size_t prefix_depth = canon_iter_depth_for(m);
  bool borrowed = vecss;
  if (!vecss) {
    size_t worker_buf_size = CHUNK * m * n_thrds;
    size_t queue_buf_size = CHUNK * prefix_depth * (1 + Q_CAP);
    vecss = malloc((worker_buf_size + queue_buf_size) * sizeof(size_t));
    assert(vecss);
  }
  *st = (proc_state_t){.mode = mode,
                       .n = n,
                       .n_args = n_args,
                       .m = m,
                       .idx = 0,
                       .np = np,
                       .ps = ps,
                       .quiet = quiet,
                       .snapshot = snapshot,
                       .n_thrds = n_thrds,
                       .vecss = vecss,
                       .borrowed = borrowed};

  source_t *src = malloc(sizeof *src);
  assert(src);
  *src = (source_t){.next = proc_next, .destroy = proc_destroy, .state = st};
  return src;
}
