#include "debug.h"
#include "source_process.h"
#include "maths.h"
#include "primes.h"
#include "mss.h"
#include "progress.h"
#include "queue.h"
#include "snapshot.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>

static uint64_t m_for(uint64_t n) {
  return 2*((n+1)/4)+1;
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
  ctx->fact_M = malloc((n+1)*sizeof(uint64_t));
  assert(ctx->fact_M);
  ctx->fact_M[0] = ctx->r;
  for (size_t i = 1; i < n+1; ++i)
    ctx->fact_M[i] = mont_mul(ctx->fact_M[i-1], ctx->nat_M[i], p, ctx->p_dash);
  ctx->fact_inv_M = malloc((n+1)*sizeof(uint64_t));
  assert(ctx->fact_inv_M);
  ctx->fact_inv_M[n] = mont_inv(ctx->fact_M[n], ctx->r, p, ctx->p_dash);
  for (size_t i = n; i; --i)
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

static uint64_t multinomial_mod_p(const prim_ctx_t *ctx, const size_t *ms, size_t len) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;

  uint64_t coeff = ctx->fact_M[ctx->n - 1];
  for (size_t i = 0; i < len; ++i)
    coeff = mont_mul(coeff, ctx->fact_inv_M[ms[i]], p, p_dash);

  return coeff;
}

static uint64_t det_mod_p(uint64_t *A, size_t dim, const prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;
  uint64_t det = ctx->r, scaling_factor = ctx->r;

  for (size_t k = 0; k < dim; ++k) {
    size_t pivot_i = k;
    while (pivot_i < dim && A[pivot_i*dim + k] == 0) ++pivot_i;
    if (pivot_i == dim) return 0;
    if (pivot_i != k) {
      for (size_t j = 0; j < dim; ++j) {
        uint64_t tmp = A[k*dim + j];
        A[k*dim + j] = A[pivot_i*dim + j];
        A[pivot_i*dim + j] = tmp;
      }
      det = p - det;
    }

    uint64_t pivot = A[k*dim + k];
    det = mont_mul(det, pivot, p, p_dash);

    for (size_t i = k + 1; i < dim; ++i) {
      scaling_factor = mont_mul(scaling_factor, pivot, p, p_dash);
      uint64_t multiplier = A[i*dim + k];
      for (size_t j = k; j < dim; ++j)
        A[i*dim + j] = mont_mul_sub(A[i*dim + j], pivot, A[k*dim + j], multiplier, p, p_dash);
    }
  }

  return mont_mul(det, mont_inv(scaling_factor, ctx->r, p, p_dash), p, p_dash);
}

static uint64_t f_fst_term(uint64_t *c, const prim_ctx_t *ctx) {
  const uint64_t m = ctx->m, p = ctx->p, p_dash = ctx->p_dash;
  uint64_t acc = ctx->r;

  for (size_t a = 0; a < m; ++a) {
    uint64_t ca = c[a];
    if (ca >= 2) {
      uint64_t base = ctx->jk_sums_M[jk_pos(a, a, m)];
      uint64_t e = (ca*(ca-1)) / 2;
      acc = mont_mul(acc, mont_pow(base, e, ctx->r, p, p_dash), p, p_dash);
    }
  }

  for (size_t a = 0; a < m; ++a) {
    uint64_t ca = c[a];
    if (!ca) continue;
    for (size_t b = a+1; b < m; ++b) {
      uint64_t cb = c[b];
      if (!cb) continue;
      acc = mont_mul(acc, mont_pow(ctx->jk_sums_M[jk_pos(a, b, m)], ca*cb, ctx->r, p, p_dash), p, p_dash);
    }
  }

  return acc;
}

static uint64_t f_snd_trm(uint64_t *c, const prim_ctx_t *ctx) {
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

static uint64_t f(uint64_t *vec, const prim_ctx_t *ctx) {
  return mont_mul(f_fst_term(vec, ctx), f_snd_trm(vec, ctx), ctx->p, ctx->p_dash);
}

typedef struct {
  uint64_t n, m, *ps;
  size_t idx, np, *vecss;
  bool quiet, snapshot;
  size_t n_thrds;
} proc_state_t;

typedef struct {
  Atomic *done;
  const prim_ctx_t *ctx;
  queue_t *q;
  size_t *vecs;
  uint64_t *l_acc, *acc;
  pthread_mutex_t *acc_mu;
  bool idle;
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

static void *residue_for_prime(void *ud) {
  worker_t *worker = ud;
  const prim_ctx_t *ctx = worker->ctx;
  uint64_t m = ctx->m, p = ctx->p, l_acc = 0;
  size_t *vecs = worker->vecs;
  worker->l_acc = &l_acc;

  for (;;) {
    size_t n_vec = queue_pop(worker->q, vecs, w_idle, worker);
    if (!n_vec) break;

    for (size_t c = 0; c < n_vec; ++c) {
      size_t *vec = &vecs[c*m];
      uint64_t f_0 = f(vec, ctx);
      uint64_t const coeff_baseline = multinomial_mod_p(ctx, vec, m);

      // Loop over each "rotation" of the vector of argument multiplicities. This is
      // equivilent to multiplying all the coefficients by w
      for (size_t r = 0; r < m; ++r) {
        // We require there always be at least one "1" in the arguments to f() (per the paper)
        // that is to say if the multiplicty of "1" arguments is zero - we should skip this case
        if (vec[r] == 0) continue;

        // The multinomial coefficient would be constant over all "rotations" of the multiplicities
        // but because we're assuming at least one argument is always "1" which requires us to subtract
        // 1 from the first multiplicity. Rather than recompute the full coeff each time we can take a
        // baseline "coefficient" and multiply it by j to convert 1/j! to 1/(j-1!)
        size_t coeff = mont_mul(coeff_baseline, ctx->nat_M[vec[r]], p, ctx->p_dash);

        size_t idx = (2*r) % m;
        uint64_t f_n = mont_mul(coeff, mont_mul(f_0, ctx->ws_M[idx ? m-idx : 0], p, ctx->p_dash), p, ctx->p_dash);
        l_acc = add_mod_u64(l_acc, f_n, p);
      }
    }
    atomic_fetch_add_explicit(worker->done, n_vec, memory_order_relaxed);
  }

  (void)w_idle(worker);
  return NULL;
}

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

static int ret(proc_state_t *st, prim_ctx_t *ctx, uint64_t acc, uint64_t *res, uint64_t *p_ret) {
  uint64_t denom = mont_inv(mont_pow(ctx->nat_M[ctx->m], ctx->n-1, ctx->r, ctx->p, ctx->p_dash), ctx->r, ctx->p, ctx->p_dash);
  uint64_t ret = mont_mul(mont_mul(acc, denom, ctx->p, ctx->p_dash), 1, ctx->p, ctx->p_dash);
  prim_ctx_free(ctx);

  *p_ret = st->ps[st->idx++];
  *res = ret;
  return 1;
}
static int proc_next(source_t *self, uint64_t *res, uint64_t *p_ret) {
  proc_state_t *st = self->state;
  if (st->idx == st->np) return 0;

  uint64_t n = st->n, m = st->m, p = st->ps[st->idx];
  uint64_t w = mth_root_mod_p(p, m);
  prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

  const size_t siz = canon_iter_size(m, n);

  Atomic done = 0;
  uint64_t acc = 0;

  uint64_t iter_st[m+5];
  size_t st_len = 0;

  if (st->snapshot)
    snapshot_try_resume(n, p, &done, &acc, iter_st, &st_len);
  assert(done <= siz);

  if (done == siz) return ret(st, ctx, acc, res, p_ret);

  progress_t prog;
  if (!st->quiet)
    progress_start(&prog, p, &done, siz);

  queue_t *q = queue_new(n, m, iter_st, st_len, &st->vecss[st->n_thrds*CHUNK*m]);

  pthread_t worker[st->n_thrds];
  worker_t w_ctxs[st->n_thrds];
  bool *idles[st->n_thrds];
  pthread_mutex_t acc_mu = PTHREAD_MUTEX_INITIALIZER;
  for (size_t i = 0; i < st->n_thrds; ++i) {
    w_ctxs[i] = (worker_t){ .ctx = ctx, .done = &done, .q = q, .vecs = &st->vecss[i*CHUNK*m], .idle = false, .acc = &acc, .acc_mu = &acc_mu };
    idles[i] = &w_ctxs[i].idle;
    pthread_create(&worker[i], NULL, residue_for_prime, &w_ctxs[i]);
  }

  snapshot_t ss;
  if (st->snapshot)
    snapshot_start(&ss, n, p, st->n_thrds, q, idles, &done, &acc);

  queue_fill(q);

  for (size_t i = 0; i < st->n_thrds; ++i)
    pthread_join(worker[i], NULL);

  queue_free(q);
  if (st->snapshot)
    snapshot_stop(&ss);

  if (!st->quiet)
    progress_stop(&prog);

  return ret(st, ctx, acc, res, p_ret);
}

static void proc_destroy(source_t *self) {
  proc_state_t *st = self->state;
  free(st->vecss);
  free(st->ps);
  free(st);
  free(self);
}

#define P_STRIDE (1ULL << 10)

source_t *source_process_new(uint64_t n, uint64_t m_id, bool quiet, bool snapshot) {
  uint64_t m = m_for(n);
  size_t np;
  assert(m_id < P_STRIDE);
  uint64_t *ps = build_prime_list(n, m, m_id, P_STRIDE, &np);

  proc_state_t *st = malloc(sizeof(*st));
  assert(st);
  size_t n_thrds = get_num_threads();
  size_t *vecss = malloc(CHUNK*m*(n_thrds+1+Q_CAP)*sizeof(size_t));
  assert(vecss);
  *st = (proc_state_t){ .n = n, .m = m, .idx = 0, .np = np, .ps = ps, .quiet = quiet, .snapshot = snapshot, .n_thrds = n_thrds, .vecss = vecss };

  source_t *src = malloc(sizeof *src);
  assert(src);
  *src = (source_t){ .next = proc_next, .destroy = proc_destroy, .state = st };
  return src;
}
