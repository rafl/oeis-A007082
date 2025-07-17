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
  return 2*((n+1)/4)+1;
}

static void create_exps(size_t *ms, size_t len, uint64_t *dst) {
  size_t idx = 0;

  for (size_t exp = 0; exp < len; ++exp) {
    size_t reps = ms[exp] - (exp == 0);
    for (size_t k = 0; k < reps; ++k)
      dst[idx++] = exp;
  }

  dst[idx] = 0;
}

typedef struct {
  uint64_t n, m, p, p_dash, r, r2, *jk_prod_M, *nat_M, *nat_inv_M, *jk_sums_M, *ws_M, *fact_M, *fact_inv_M;
} prim_ctx_t;

static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  return j*m + k;
}

static prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t m, uint64_t p, uint64_t w) {
  prim_ctx_t *ctx = malloc(sizeof(prim_ctx_t));
  assert(ctx);
  ctx->n = n;
  ctx->m = m;
  ctx->p = p;
  ctx->p_dash = (uint64_t)(-inv64_u64(p));
  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;

  ctx->ws_M = malloc(m*sizeof(uint64_t));
  assert(ctx->ws_M);
  ctx->ws_M[0] = ctx->r;
  ctx->ws_M[1] = mont_mul(w, ctx->r2, p, ctx->p_dash);
  for (size_t i = 2; i < m; ++i)
    ctx->ws_M[i] = mont_mul(ctx->ws_M[i-1], ctx->ws_M[1], p, ctx->p_dash);
  uint64_t jk_pairs_M[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_pairs_M[jk_pos(j, k, m)] = mont_mul(ctx->ws_M[j], ctx->ws_M[k ? m-k : 0], p, ctx->p_dash);
  }
  ctx->jk_sums_M = malloc(m*m*sizeof(uint64_t));
  assert(ctx->jk_sums_M);
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      ctx->jk_sums_M[jk_pos(j, k, m)] =
        add_mod_u64(jk_pairs_M[jk_pos(j, k, m)], jk_pairs_M[jk_pos(k, j, m)], p);
  }
  ctx->jk_prod_M = malloc(m*m*sizeof(uint64_t));
  assert(ctx->jk_prod_M);
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      size_t pos = jk_pos(j, k, m);
      uint64_t sum_inv = mont_inv(ctx->jk_sums_M[pos], ctx->r, p, ctx->p_dash);
      ctx->jk_prod_M[pos] = mont_mul(jk_pairs_M[pos], sum_inv, p, ctx->p_dash);
    }
  }
  ctx->nat_M = malloc((n+1)*sizeof(uint64_t));
  assert(ctx->nat_M);
  for (size_t i = 0; i <= n; ++i)
    ctx->nat_M[i] = mont_mul(i, ctx->r2, p, ctx->p_dash);
  ctx->nat_inv_M = malloc((n + 1) * sizeof(uint64_t));
  assert(ctx->nat_inv_M);
  ctx->nat_inv_M[0] = 0;
  for (size_t k = 1; k <= n; ++k)
    ctx->nat_inv_M[k] = mont_inv(ctx->nat_M[k], ctx->r, p, ctx->p_dash);
  ctx->fact_M = malloc(n*sizeof(uint64_t));
  assert(ctx->fact_M);
  ctx->fact_M[0] = ctx->r;
  for (size_t i = 1; i < n; ++i)
    ctx->fact_M[i] = mont_mul(ctx->fact_M[i-1], ctx->nat_M[i], p, ctx->p_dash);
  ctx->fact_inv_M = malloc(n*sizeof(uint64_t));
  assert(ctx->fact_inv_M);
  ctx->fact_inv_M[n-1] = mont_inv(ctx->fact_M[n-1], ctx->r, p, ctx->p_dash);
  for (size_t i = n-1; i; --i)
    ctx->fact_inv_M[i-1] = mont_mul(ctx->fact_inv_M[i], ctx->nat_M[i], p, ctx->p_dash);

  return ctx;
}

static void prim_ctx_free(prim_ctx_t *ctx) {
  free(ctx->fact_inv_M);
  free(ctx->fact_M);
  free(ctx->ws_M);
  free(ctx->jk_sums_M);
  free(ctx->nat_inv_M);
  free(ctx->nat_M);
  free(ctx->jk_prod_M);
  free(ctx);
}

static uint64_t multinomial_mod_p(prim_ctx_t *ctx, const size_t *ms, size_t len) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;

  size_t tot = 0;
  for (size_t i = 0; i < len; ++i)
    tot += ms[i] - (i == 0);

  uint64_t coeff = ctx->fact_M[tot];
  for (size_t i = 0; i < len; ++i)
    coeff = mont_mul(coeff, ctx->fact_inv_M[ms[i] - (i == 0)], p, p_dash);

  return coeff;
}

static uint64_t det_mod_p(uint64_t *A, size_t dim, prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;
  uint64_t det = ctx->r;

  for (size_t k = 0; k < dim; ++k) {
    uint64_t pivot = A[k*dim + k];
    uint64_t inv_pivot = mont_inv(pivot, ctx->r, p, p_dash);
    det = mont_mul(det, pivot, p, p_dash);

    for (size_t i = k + 1; i < dim; ++i) {
      uint64_t factor = mont_mul(A[i*dim + k], inv_pivot, p, p_dash);
      for (size_t j = k; j < dim; ++j) {
        uint64_t tmp = mont_mul(factor, A[k*dim + j], p, p_dash);
        A[i*dim + j] = sub_mod_u64(A[i*dim + j], tmp, p);
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
  return acc;
}

static uint64_t f_snd_trm(uint64_t *c, prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, m = ctx->m;

  // active groups
  size_t typ[m], r = 0;
  for (size_t i = 0; i < m; ++i) {
    if (c[i]) {
      typ[r] = i;
      ++r;
    }
  }

  uint64_t prod_M = ctx->r;
  for (size_t a = 0; a < r; ++a) {
    size_t i = typ[a];
    uint64_t sum = 0;
    for (size_t b = 0; b < r; ++b) {
      size_t j = typ[b];
      uint64_t w = ctx->jk_prod_M[jk_pos(i, j, m)];
      sum = add_mod_u64(sum, mont_mul(ctx->nat_M[c[j]], w, p, ctx->p_dash), p);
    }

    prod_M = mont_mul(prod_M, mont_pow(sum, c[i]-1, ctx->r, p, ctx->p_dash), p, ctx->p_dash);
  }

  prod_M = mont_mul(prod_M, ctx->nat_inv_M[c[0]], p, ctx->p_dash);

  // quotient minor
  size_t dim = r - 1;
  if (!dim)
    return prod_M;

  uint64_t A[dim*dim];
  for (size_t a = 1; a < r; ++a) {
    size_t i = typ[a];

    // contribution from the deleted block
    uint64_t w_del = ctx->jk_prod_M[jk_pos(i, 0, m)];
    uint64_t diag = mont_mul(ctx->nat_M[c[0]], w_del, p, ctx->p_dash);

    // remaining off-diagonal blocks
    for (size_t b = 1; b < r; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      uint64_t w = ctx->jk_prod_M[jk_pos(i, j, m)];
      uint64_t v = mont_mul(ctx->nat_M[c[j]], w, p, ctx->p_dash);

      A[(a-1)*dim + (b-1)] = p - v;
      diag = add_mod_u64(diag, v, p);
    }

    A[(a-1)*dim + (a-1)] = diag;
  }

  return mont_mul(prod_M, det_mod_p(A, dim, ctx), p, ctx->p_dash);
}

static uint64_t f(uint64_t *vec, uint64_t *exps, prim_ctx_t *ctx) {
  return mont_mul(f_fst_term(exps, ctx), f_snd_trm(vec, ctx), ctx->p, ctx->p_dash);
}

typedef struct {
  uint64_t n, m, *ps;
  size_t idx, np;
  bool quiet;
} proc_state_t;

static inline void canon_iter_skip(canon_iter_t *it, size_t k, size_t *vec) {
  while (k--) canon_iter_next(it, vec);
}

static uint64_t residue_for_prime(bool quiet, uint64_t n, uint64_t m, uint64_t p) {
  uint64_t w = mth_root_mod_p(p, m);
  prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

  const size_t siz = canon_iter_size(m, n);

  _Atomic size_t done = 0;
  progress_t prog;
  if (!quiet)
    progress_start(&prog, p, &done, siz);

  _Atomic size_t next_idx = 0;
  #define CHUNK 4096

  uint64_t acc = 0;
  #pragma omp parallel
  {
    uint64_t exps[n], l_acc = 0;
    size_t vec[m], scratch[m+1];
    canon_iter_t can_it;

    canon_iter_new(&can_it, m, n, scratch);
    size_t l_rank = 0;

    for (;;) {
      size_t start = atomic_fetch_add_explicit(&next_idx, CHUNK, memory_order_relaxed);
      if (start >= siz) break;
      size_t delta = start - l_rank;
      canon_iter_skip(&can_it, delta, vec);

      size_t remaining = siz - start;
      size_t block     = remaining < CHUNK ? remaining : CHUNK;
      l_rank = start;

      for (size_t c = 0; c < block; ++c) {
        canon_iter_next(&can_it, vec);
        create_exps(vec, m, exps);
        uint64_t f_0 = f(vec, exps, ctx);
        size_t vec_rots[2*m];
        memcpy(vec_rots, vec, m*sizeof(uint64_t));
        memcpy(vec_rots+m, vec_rots, m*sizeof(uint64_t));
        for (size_t r = 0; r < m; ++r) {
          size_t *vec_r = vec_rots + r;
          if (vec_r[0] == 0) continue;
          uint64_t coeff = multinomial_mod_p(ctx, vec_r, m);
          size_t idx = (2*r) % m;
          uint64_t f_n = mont_mul(coeff, mont_mul(f_0, ctx->ws_M[idx ? m-idx : 0], p, ctx->p_dash), p, ctx->p_dash);
          l_acc = add_mod_u64(l_acc, f_n, p);
        }
      }
      atomic_fetch_add_explicit(&done, block, memory_order_relaxed);
      l_rank += block;
    }

    #pragma omp critical
    acc = add_mod_u64(acc, l_acc, p);
  }

  if (!quiet)
    progress_stop(&prog);
  uint64_t denom = mont_inv(mont_pow(ctx->nat_M[m], n-1, ctx->r, p, ctx->p_dash), ctx->r, p, ctx->p_dash);
  uint64_t ret = mont_mul(mont_mul(acc, denom, ctx->p, ctx->p_dash), 1, ctx->p, ctx->p_dash);
  prim_ctx_free(ctx);
  return ret;
}

static int proc_next(source_t *self, uint64_t *res, uint64_t *p) {
  proc_state_t *st = self->state;
  if (st->idx == st->np) return 0;

  *p = st->ps[st->idx++];
  *res = residue_for_prime(st->quiet, st->n, st->m, *p);
  return 1;
}

static void proc_destroy(source_t *self) {
  proc_state_t *st = self->state;
  free(st->ps);
  free(st);
  free(self);
}

#define P_STRIDE (1ULL << 10)

source_t *source_process_new(uint64_t n, uint64_t m_id, bool quiet) {
  uint64_t m = m_for(n);
  size_t np;
  assert(m_id < P_STRIDE);
  uint64_t *ps = build_prime_list(n, m, m_id, P_STRIDE, &np);

  proc_state_t *st = malloc(sizeof(*st));
  assert(st);
  *st = (proc_state_t){ .n = n, .m = m, .idx = 0, .np = np, .ps = ps, .quiet = quiet };

  source_t *src = malloc(sizeof *src);
  assert(src);
  *src = (source_t){ .next = proc_next, .destroy = proc_destroy, .state = st };
  return src;
}
