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

static void create_exps(size_t *ms, // "necklace normalzed" coefficients
  size_t len, 
  uint64_t *dst) {
  size_t idx = 0;

  for (size_t exp = 0; exp < len; ++exp) {
    // number of times we're repeating a given coeff is ms[exp] - unless exp == for some reasons
    // and then we subtract 1?
    // Oh because we always have 1 as one of the args? Hmmm ok
    // I guess this is why we skip the zero case too?
    size_t reps = ms[exp] - (exp == 0);
    for (size_t k = 0; k < reps; ++k)
      dst[idx++] = exp; // So we're turning 1 3 7 into 1 2 2 2 3 3 3 3 3 3 3 ( except for the skip the first 1 thing)
  }

  dst[idx] = 0;
}

// precomputation of shared forms
// "montgomery form" (do some bitshifts and then a mult instead of a divide)
// That's in maths.c
typedef struct {
  uint64_t n, m, p, //n = number of veritcies //obvious
  p_dash, r, r2, // montgomery stuff _M implies something is in montgomery form
  *jk_prod_M, // cache of w^j*w^-k / (w^-j*w^k + w^j*w^-k)
  *nat_M, // natural numbers up to n (inclusive)
  *nat_inv_M, // inverses of natural numbers up to n (inclusive)
  *jk_sums_M, // w^-j*w^k + w^j*w^-k 
  *ws_M, // powers of omega (m form)
  *fact_M, // i! for i <= n
  *fact_inv_M; // 1/i! for i <= n
} prim_ctx_t;

// index into rectanular array
static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  return j*m + k;
}

// shared over threads - not mutated
static prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t m, uint64_t p, uint64_t w) {
  prim_ctx_t *ctx = malloc(sizeof(prim_ctx_t));
  assert(ctx);
  ctx->n = n;
  ctx->m = m;
  ctx->p = p;
  ctx->p_dash = (uint64_t)(-inv64_u64(p));
  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;

  // initialize roots of unity
  ctx->ws_M = malloc(m*sizeof(uint64_t));
  assert(ctx->ws_M);
  ctx->ws_M[0] = ctx->r;
  ctx->ws_M[1] = mont_mul(w, ctx->r2, p, ctx->p_dash);
  for (size_t i = 2; i < m; ++i)
    ctx->ws_M[i] = mont_mul(ctx->ws_M[i-1], ctx->ws_M[1], p, ctx->p_dash);

  // w^j * w^-k lookup - not actually inserted into the context
  uint64_t jk_pairs_M[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_pairs_M[jk_pos(j, k, m)] = mont_mul(ctx->ws_M[j], ctx->ws_M[k ? m-k : 0], p, ctx->p_dash);
  }

  // cache of // w^-j*w^k + w^j*w^-k 
  ctx->jk_sums_M = malloc(m*m*sizeof(uint64_t));
  assert(ctx->jk_sums_M);
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      ctx->jk_sums_M[jk_pos(j, k, m)] =
        add_mod_u64(jk_pairs_M[jk_pos(j, k, m)], jk_pairs_M[jk_pos(k, j, m)], p);
  }

  // cache of w^j*w^-k / (w^-j*w^k + w^j*w^-k)
  ctx->jk_prod_M = malloc(m*m*sizeof(uint64_t));
  assert(ctx->jk_prod_M);
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      size_t pos = jk_pos(j, k, m);
      uint64_t sum_inv = mont_inv(ctx->jk_sums_M[pos], ctx->r, p, ctx->p_dash);
      ctx->jk_prod_M[pos] = mont_mul(jk_pairs_M[pos], sum_inv, p, ctx->p_dash);
    }
  }

  // 1 to n
  ctx->nat_M = malloc((n+1)*sizeof(uint64_t));
  assert(ctx->nat_M);
  for (size_t i = 0; i <= n; ++i)
    ctx->nat_M[i] = mont_mul(i, ctx->r2, p, ctx->p_dash);

  // 1/i for i = 1 to n
  ctx->nat_inv_M = malloc((n + 1) * sizeof(uint64_t));
  assert(ctx->nat_inv_M);
  ctx->nat_inv_M[0] = 0;
  for (size_t k = 1; k <= n; ++k)
    ctx->nat_inv_M[k] = mont_inv(ctx->nat_M[k], ctx->r, p, ctx->p_dash);

  // i! for i = 1 to n
  ctx->fact_M = malloc(n*sizeof(uint64_t));
  assert(ctx->fact_M);
  ctx->fact_M[0] = ctx->r;
  for (size_t i = 1; i < n; ++i)
    ctx->fact_M[i] = mont_mul(ctx->fact_M[i-1], ctx->nat_M[i], p, ctx->p_dash);

  // 1/i! for i = 1 to n
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

// Calculate the multinomal coefficient where the powers of x_i are given by *ms
static uint64_t multinomial_mod_p(const prim_ctx_t *ctx, const size_t *ms, size_t len) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;

  size_t tot = 0;
  // is this total not just n - 1?
  for (size_t i = 0; i < len; ++i)
    tot += ms[i] - (i == 0);

  uint64_t coeff = ctx->fact_M[tot];
  for (size_t i = 0; i < len; ++i)
    coeff = mont_mul(coeff, ctx->fact_inv_M[ms[i] - (i == 0)], p, p_dash);

  return coeff;
}

// TODO - understand this better - could this be being computed more efficiently?
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

// The full product over all pairwise combinations
static uint64_t f_fst_term(uint64_t *exps, const prim_ctx_t *ctx) {
  uint64_t acc = ctx->r; // r is basically 1 in mont form
  for (size_t j = 0; j < ctx->n; ++j) {
    for (size_t k = j + 1; k < ctx->n; ++k) {
      uint64_t t = ctx->jk_sums_M[jk_pos(exps[j], exps[k], ctx->m)];
      acc = mont_mul(acc, t, ctx->p, ctx->p_dash);
    }
  }
  return acc;
}

static uint64_t f_snd_trm(uint64_t *c, const prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, m = ctx->m;

  // active groups
  // This is the indexs of the non zero elements of c
  // Which is also the powers of w they correspond to
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

    // This is basically the column sum of the ith column of a
    uint64_t sum = 0;
    // for each non zero power of omega w^j in our args
    for (size_t b = 0; b < r; ++b) {
      size_t j = typ[b];
      // w^i*w^-j
      // (n.b. we could distribute over w^i here if we wanted)
      uint64_t w = ctx->jk_prod_M[jk_pos(i, j, m)];

      // sum += w * multiplicity of (w^j)
      sum = add_mod_u64(sum, mont_mul(ctx->nat_M[c[j]], w, p, ctx->p_dash), p);
    }

    prod_M = mont_mul(prod_M, mont_pow(sum, c[i]-1, ctx->r, p, ctx->p_dash), p, ctx->p_dash);

    // So like for each column we "delete" we multiply by the column sum...
  }

  // Prod M = product[a = 0->r-1] sum[b = 0->1-r] w^coeff_cnt[a] * w^-coeff_cnt[b]

  prod_M = mont_mul(prod_M, ctx->nat_inv_M[c[0]], p, ctx->p_dash);
  // We divide by the multiplicty of 1??

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

  // I guess prod_M is some magic compensation coefficient
  return mont_mul(prod_M, det_mod_p(A, dim, ctx), p, ctx->p_dash);
}

// Exps = exponents of powers of unity [0, 1, 1, 2, 4] -> [1, w, w, w^2, w^4]
// Vec is staying in count space (so would be ) [1, 2, 1, 0, 1] for above
static uint64_t f(uint64_t *vec, uint64_t *exps, const prim_ctx_t *ctx) {
  return mont_mul(f_fst_term(exps, ctx), f_snd_trm(vec, ctx), ctx->p, ctx->p_dash);
}

// state of whole process (shared over threads)
typedef struct {
  uint64_t n, /* element of seq */ m, /*w is the mth root of unity*/ *ps /* list of primes */;
  size_t idx /* withth prime*/, np /* number of primes*/, *vecss /*Some buffers for vectors*/;
  bool quiet, snapshot; /*mode stuff - saving snapshots so you can restart*/
  size_t n_thrds; /*number of threads*/
} proc_state_t;

typedef struct {
  _Atomic size_t *done;
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

// thread function for doing work on the "maths stuff"
static void *residue_for_prime(void *ud) {
  worker_t *worker = ud;
  const prim_ctx_t *ctx = worker->ctx;
  uint64_t n = ctx->n, m = ctx->m, p = ctx->p;
  uint64_t exps[n], /* what are you?*/ l_acc = 0; // l_acc is where we're going to accumulate the total residual from the stuff we pull from our work queue
  size_t *vecs = worker->vecs;
  worker->l_acc = &l_acc;

  for (;;) {
    size_t n_vec = queue_pop(worker->q, vecs, w_idle, worker);
    if (!n_vec) break;

    for (size_t c = 0; c < n_vec; ++c) {
      // each vector has len m
      // each vector contains "necklace normalized" coefficient counts
      // i.e. 1 3 7 = 1 lot of w^1, 3 lots of w^2, 7 lots of w^3
      size_t *vec = &vecs[c*m];
      create_exps(vec, m, exps);
      // This is the full f from the paper
      // We evaluate it in two halfs (the product and the matrix determinant from theorem 2)
      uint64_t f_0 = f(vec, exps, ctx);
      size_t vec_rots[2*m];
      memcpy(vec_rots, vec, m*sizeof(uint64_t));
      memcpy(vec_rots+m, vec_rots, m*sizeof(uint64_t));
      // go over each "rotation" of the vector - this is the same as multiply by omega
      // e.g. go from f_0 = f(1 3 7 0) to f_1(0 1 3 7) is the equivient of multiplying each coeffecient
      // by omega
      // we index in reverse to make this the same as multiplying by w
      for (size_t r = 0; r < m; ++r) {
        size_t *vec_r = vec_rots + r;
        // We can skip this because... I am not sure
        // Because we always have at least one 1 in our calc I guess...
        if (vec_r[0] == 0) continue;
        // TODO - why do we compute this for each loop - isn't this a constant for a given set of coeffs?
        // Are we messing around with this a bit because of the constant 1 at the end?
        uint64_t coeff = multinomial_mod_p(ctx, vec_r, m);
        size_t idx = (2*r) % m;
        // coeff * f_0 * w^(m-idx) = coeff * f_0 * w^-2r
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

// Implementation of the virtual function - entrypoint for calculation
static int proc_next(source_t *self, uint64_t *res, uint64_t *p_ret) {
  proc_state_t const *st = self->state;
  if (st->idx == st->np) return 0;

  // n, m, prime
  uint64_t n = st->n, m = st->m, p = st->ps[st->idx];
  // root of unity
  uint64_t w = mth_root_mod_p(p, m);
  prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

  const size_t siz = canon_iter_size(m, n);

  _Atomic size_t done = 0;
  uint64_t acc = 0;

  uint64_t iter_st[m+5];
  size_t st_len = 0;

  if (st->snapshot)
    snapshot_try_resume(n, p, &done, &acc, iter_st, &st_len);
  assert(done <= siz);

  if (done == siz) return ret(st, ctx, acc, res, p_ret);

  progress_t prog;
  if (!st->quiet)
  // progress bar stuff
    progress_start(&prog, p, &done, siz);

  // shared work queue
  queue_t *q = queue_new(n, m, iter_st, st_len, &st->vecss[st->n_thrds*CHUNK*m]);

  // make some threads
  pthread_t worker[st->n_thrds];
  worker_t w_ctxs[st->n_thrds];
  bool *idles[st->n_thrds];
  pthread_mutex_t acc_mu = PTHREAD_MUTEX_INITIALIZER;
  for (size_t i = 0; i < st->n_thrds; ++i) {
    w_ctxs[i] = (worker_t){ .ctx = ctx, .done = &done, .q = q, .vecs = &st->vecss[i*CHUNK*m], .idle = false, .acc = &acc, .acc_mu = &acc_mu };
    idles[i] = &w_ctxs[i].idle;
    // worker threads
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
