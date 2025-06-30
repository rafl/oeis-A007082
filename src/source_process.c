#include "debug.h"
#include "source_process.h"
#include "maths.h"
#include "primes.h"
#include "mss.h"
#include "progress.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

static uint64_t m_for(uint64_t n) {
  uint64_t x = (n+1)/2;
  if (x & 1) return x;
  return (n+3)/2;
}

static void create_exps(size_t *ms, size_t len, uint64_t *dst) {
  size_t idx = 0;

  for (size_t exp = 0; exp < len; ++exp) {
    for (size_t k = 0; k < ms[exp]; ++k)
      dst[idx++] = exp;
  }

  dst[idx] = 0;
}

typedef struct {
  uint64_t n, m, p, p_dash, r, r2, *jk_prod_M, *jk_prod, *nat_M, *jk_sums_M, *ws;
  size_t *binoms;
} prim_ctx_t;

static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  return j*m + k;
}

static prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t m, uint64_t p, uint64_t w) {
  prim_ctx_t *ctx = malloc(sizeof(prim_ctx_t));
  ctx->n = n;
  ctx->m = m;
  ctx->p = p;
  ctx->p_dash = (uint64_t)(0 - inv64_u64(p));
  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;

  // TODO: move all of the locals into M-space for neatness. wouldn't really
  // affect total run-time cause it's all per-prime only.
  ctx->ws = malloc(m*sizeof(uint64_t));
  for (size_t i = 0; i < m; ++i)
    ctx->ws[i] = pow_mod_u64(w, i, p);
  uint64_t ws_inv[m];
  for (size_t i = 0; i < m; ++i)
    ws_inv[i] = inv_mod_u64(ctx->ws[i], p);
  uint64_t jk_pairs[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_pairs[jk_pos(j, k, m)] = mul_mod_u64(ctx->ws[j], ws_inv[k], p);
  }
  uint64_t jk_sums[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_sums[jk_pos(j, k, m)] =
        add_mod_u64(jk_pairs[jk_pos(j, k, m)], jk_pairs[jk_pos(k, j, m)], p);
  }
  ctx->jk_prod = malloc(m*m*sizeof(uint64_t));
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      size_t pos = jk_pos(j, k, m);
      ctx->jk_prod[pos] = mul_mod_u64(jk_pairs[pos], inv_mod_u64(jk_sums[pos], p), p);
    }
  }
  ctx->jk_prod_M = malloc(m*m*sizeof(uint64_t));
  for (size_t i = 0; i < m*m; ++i)
    ctx->jk_prod_M[i] = mont_mul(ctx->jk_prod[i], ctx->r2, p, ctx->p_dash);
  ctx->nat_M = malloc((n+1)*sizeof(uint64_t));
  // nat_M[0] is intentionally uninitialised to maximise developer engagement
  // when that inevitably backfires. index is off by one to avoid extra maths
  // during look-ups, which might be a premature optimisation. the extra word
  // of memory doesn't matter.
  for (size_t i = 1; i <= n; ++i)
    ctx->nat_M[i] = mont_mul(i, ctx->r2, p, ctx->p_dash);
  ctx->jk_sums_M = malloc(m*m*sizeof(uint64_t));
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      size_t pos = jk_pos(j,k,m);
      ctx->jk_sums_M[pos] = mont_mul(jk_sums[pos], ctx->r2, p, ctx->p_dash);
    }
  }
  size_t rows = (n-1) + m + 1, cols = m+1;
  ctx->binoms = malloc(rows*cols*sizeof(size_t));
  for (size_t j = 0; j < rows; ++j) {
    for (size_t k = 0; k < cols; ++k) {
      size_t v = (k == 0 || k == j) ? 1
               : (k > j) ? 0
               : ctx->binoms[(j-1)*cols + (k-1)] + ctx->binoms[(j-1)*cols + k];
      ctx->binoms[j*cols + k] = v;
    }
  }

  return ctx;
}

static inline size_t binom_pos(size_t j, size_t k, size_t m) {
  return j*(m+1) + k;
}

static void prim_ctx_free(prim_ctx_t *ctx) {
  free(ctx->ws);
  free(ctx->binoms);
  free(ctx->jk_sums_M);
  free(ctx->nat_M);
  free(ctx->jk_prod_M);
  free(ctx->jk_prod);
  free(ctx);
}

static uint64_t multinomial_mod_p(prim_ctx_t *ctx, const size_t *ms, size_t len) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash, r2 = ctx->r2, *natM = ctx->nat_M, r = ctx->r;
  uint64_t acc = r, n = 0;

  for (size_t i = 0; i < len; ++i) {
    uint64_t d = r;

    for (size_t k = 1; k <= ms[i]; ++k) {
      acc = mont_mul(acc, natM[++n], p, p_dash);
      d = mont_mul(d, natM[k], p, p_dash);
    }

    uint64_t dinv = mont_mul(inv_mod_u64(mont_mul(d, 1, p, p_dash), p), r2, p, p_dash);
    acc = mont_mul(acc, dinv, p, p_dash);
  }

  uint64_t ret = mont_mul(acc, 1, p, p_dash);
  return ret;
}

static uint64_t det_mod_p(uint64_t *A, size_t dim, prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash, r2 = ctx->r2;
  uint64_t det = ctx->r;

  for (size_t k = 0; k < dim; ++k) {
    size_t pivot = k;
    while (pivot < dim && A[pivot*dim + k] == 0) ++pivot;
    if (pivot == dim) return 0;
    if (pivot != k) {
      for (size_t j = 0; j < dim; ++j) {
        uint64_t tmp = A[k*dim + j];
        A[k*dim + j] = A[pivot*dim + j];
        A[pivot*dim + j] = tmp;
      }
      det = p - det;
    }

    // eh whatever it's only like 20ish of these
    // TODO: check to see if maybe we can compute some inverses outside of montgomery space upfront?
    uint64_t inv_pivot = mont_mul(inv_mod_u64(mont_mul(A[k*dim + k], 1, p, p_dash), p), r2, p, p_dash);
    det = mont_mul(det, A[k*dim + k], p, p_dash);

    for (size_t i = k + 1; i < dim; ++i) {
      uint64_t factor = mont_mul(A[i*dim + k], inv_pivot, p, p_dash);
      if (factor == 0) continue;

      for (size_t j = k; j < dim; ++j) {
        uint64_t tmp = mont_mul(factor, A[k*dim + j], p, p_dash);
        uint64_t val = (A[i*dim + j] >= tmp)
                     ? A[i*dim + j] - tmp
                     : A[i*dim + j] + p - tmp;
        A[i*dim + j] = val;
      }
    }
  }

  return det;
}

static uint64_t f_fst_term(uint64_t *exps, prim_ctx_t *ctx) {
  uint64_t acc = ctx->r;
  for (size_t j = 0; j < ctx->n; ++j) {
    for (size_t k = j + 1; k < ctx->n; ++k) {
      uint64_t t = ctx->jk_sums_M[jk_pos(exps[j], exps[k], ctx->m)];
      acc = mont_mul(acc, t, ctx->p, ctx->p_dash);
    }
  }
  uint64_t ret = mont_mul(acc, 1, ctx->p, ctx->p_dash);
  return ret;
}

// TODO: move back into montgomery domain?
static uint64_t f_snd_trm(uint64_t *vec, prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, m = ctx->m;

  size_t c[m];
  // TODO: avoid copy? could adjust vec representation or handle vec[0] special below
  memcpy(c, vec, m*sizeof(size_t));
  ++c[0];

  // active groups
  size_t typ[m], r = 0, del_i = 0;
  for (size_t i = 0; i < m; ++i) {
    if (c[i]) {
      typ[r] = i;
      ++r;
    }
  }

  uint64_t prod_int = 1;
  for (size_t a = 0; a < r; ++a) {
    size_t i = typ[a];
    uint64_t sum = 0;
    for (size_t b = 0; b < r; ++b) {
      size_t j = typ[b];
      uint64_t w = ctx->jk_prod[jk_pos(i, j, m)];
      sum = add_mod_u64(sum, mul_mod_u64(c[j], w, p), p);
    }
    uint64_t d_i = sub_mod_u64(sum, ctx->jk_prod[jk_pos(i, i, m)], p);
    uint64_t lam_i = add_mod_u64(d_i, ctx->jk_prod[jk_pos(i, i, m)], p);

    if (c[i] > 1)
      prod_int = mul_mod_u64(prod_int, pow_mod_u64(lam_i, c[i] - 1, p), p);
  }

  // quotient minor
  size_t dim = r - 1;
  uint64_t A[dim ? dim*dim : 1];

  if (dim) {
    size_t row = 0;
    for (size_t a = 0; a < r; ++a) {
      size_t i = typ[a];
      if (i == del_i) continue;

      uint64_t diag = 0;
      size_t col = 0;

      // contribution from the deleted block
      uint64_t w_del = ctx->jk_prod[jk_pos(i, del_i, m)];
      uint64_t val = mul_mod_u64(c[del_i], w_del, p);
      diag = add_mod_u64(diag, val, p);

      // remaining off-diagonal blocks
      for (size_t b = 0; b < r; ++b) {
        size_t j = typ[b];
        if (j == del_i || j == i)
          continue;

        if (col == row)
          ++col;

        uint64_t w = ctx->jk_prod[jk_pos(i, j, m)];
        uint64_t v = mul_mod_u64(c[j], w, p);

        A[row*dim + col] = v ? p - v : 0;
        diag = add_mod_u64(diag, v, p);
        ++col;
      }

      A[row*dim + row] = diag;
      ++row;
    }
  }
  for (size_t i = 0; i < dim*dim; ++i)
    A[i] = mont_mul(A[i], ctx->r2, p, ctx->p_dash);
  uint64_t det_q = dim ? mont_mul(det_mod_p(A, dim, ctx), 1, p, ctx->p_dash) : 1;
  det_q = mul_mod_u64(det_q, inv_mod_u64(c[del_i], p), p);
  uint64_t ret = mul_mod_u64(prod_int, det_q, p);
  return ret;
}

static uint64_t f(uint64_t *vec, uint64_t *exps, prim_ctx_t *ctx) {
  return mul_mod_u64(f_fst_term(exps, ctx), f_snd_trm(vec, ctx), ctx->p);
}

typedef struct {
  uint64_t n, m, *ps;
  size_t idx, np;
} proc_state_t;

void rot_vec(size_t *dst, size_t *src, size_t n, size_t m) {
  assert(m > 1);
  assert(n < m);

  if (n == 0) {
    memcpy(dst, src, m * sizeof(size_t));
    return;
  }

  memcpy(dst, src + m - n, n * sizeof(size_t));
  memcpy(dst + n, src, (m-n) * sizeof(size_t));
}

static uint64_t residue_for_prime(uint64_t n, uint64_t m, uint64_t p) {
  uint64_t w = mth_root_mod_p(p, m);
  prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

  const size_t siz = canon_iter_size(m, n);

  _Atomic size_t done = 0;
  progress_t prog;
  progress_start(&prog, &done, siz);

  uint64_t acc = 0;
//  #pragma omp parallel
  {
    uint64_t exps[n], l_acc = 0;
    size_t vec[m], vec_r[m], scratch[m+1];
    canon_iter_t can_it;

    canon_iter_new(&can_it, m, n, vec, scratch);
    while (canon_iter_next(&can_it)) {
      memcpy(vec_r, vec, m*sizeof(size_t));
      --vec_r[0];
      create_exps(vec_r, m, exps);
      uint64_t f_0 = f(vec_r, exps, ctx);
      ++vec_r[0];
      for (size_t r = 0; r < m; ++r) {
        rot_vec(vec_r, vec, r, m);
        if (vec_r[0] == 0) continue;
        --vec_r[0];
        uint64_t coeff = multinomial_mod_p(ctx, vec_r, m);
        uint64_t f_n = mul_mod_u64(coeff, mul_mod_u64(f_0, ctx->ws[(2*r)%m], p), p);
        l_acc = add_mod_u64(l_acc, f_n, p);
      }
      atomic_fetch_add_explicit(&done, 1, memory_order_relaxed);
    }

//    #pragma omp critical
    acc = add_mod_u64(acc, l_acc, p);
  }

  progress_stop(&prog);
  prim_ctx_free(ctx);
  uint64_t denom = pow_mod_u64(pow_mod_u64(m % p, n - 1, p), p - 2, p);
  return mul_mod_u64(acc, denom, p);
}

static int proc_next(source_t *self, uint64_t *res, uint64_t *p) {
  proc_state_t *st = self->state;
  if (st->idx == st->np) return 0;

  *p = st->ps[st->idx++];
  *res = residue_for_prime(st->n, st->m, *p);
  return 1;
}

static void proc_destroy(source_t *self) {
  proc_state_t *st = self->state;
  free(st->ps);
  free(st);
  free(self);
}

#define P_STRIDE (1ULL << 10)

source_t *source_process_new(uint64_t n, uint64_t m_id) {
  uint64_t m = m_for(n);
  size_t np;
  assert(m_id < P_STRIDE);
  uint64_t *ps = build_prime_list(n, m, m_id, P_STRIDE, &np);

  proc_state_t *st = malloc(sizeof(*st));
  *st = (proc_state_t){ .n = n, .m = m, .idx = 0, .np = np, .ps = ps };

  source_t *src = malloc(sizeof *src);
  *src = (source_t){ .next = proc_next, .destroy = proc_destroy, .state = st };
  return src;
}
