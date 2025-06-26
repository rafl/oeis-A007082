#include "oeis.h"
#include "maths.h"

#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <gmp.h>
#include <math.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>

uint64_t prime_congruent_1_mod_m(uint64_t start, uint64_t m) {
  mpz_t z;
  mpz_init(z);

  uint64_t p = start + (m - (start % m) + 1) % m;

  while (1) {
    mpz_set_ui(z, p);
    if (mpz_probab_prime_p(z, 500)) {
      mpz_clear(z);
      return p;
    }
    p += m;
  }
}

uint64_t mth_root_mod_p(uint64_t p, uint64_t m)
{
  uint64_t phi = p - 1;
  assert(!(phi % m));

  uint64_t pf[8];
  size_t k = 0;
  factor_u64(m, pf, sizeof(pf)/sizeof(pf[0]), &k);

  uint64_t e = phi / m;
  for (uint64_t g = 2; ; ++g) {
    if (pow_mod_u64(g, phi, p) != 1) continue;

    uint64_t cand = pow_mod_u64(g, e, p);
    if (cand == 1) continue;

    int ok = 1;
    for (size_t i = 0; i < k; ++i) {
      if (pow_mod_u64(cand, m / pf[i], p) == 1) {
        ok = 0;
        break;
      }
    }
    if (ok) return cand;
  }
}

typedef struct {
  size_t n, tot, lvl, *vec, *scratch;
  bool fin;
} mss_iter_t;

void mss_iter_new(mss_iter_t *const it, size_t n, size_t r, size_t *vec, size_t *scratch) {
  it->n = n;
  it->tot = r;
  it->vec = vec;
  it->scratch = scratch;

  it->lvl = 0;
  it->fin = false;
  memset(vec, 0, n*sizeof(size_t));
  memset(scratch, 0, n*sizeof(size_t));
}

static inline void
mss_iter_init_at(mss_iter_t *const it, size_t n, size_t r, size_t *vec, size_t *scratch)
{
  it->n = n;
  it->tot = r;
  it->vec = vec;
  it->scratch = scratch;
  it->lvl = 1;
  it->fin = false;

  size_t rem = 0;
  for (size_t i = n; i-- > 0; ) {
    rem += vec[i];
    scratch[i] = rem;
  }
}

size_t mss_iter_size(size_t m, size_t n) {
  uint128_t numerator = 1;
  uint128_t denominator = 1;
  for (size_t i = 1; i < m; ++i) {
    int num = i + n - 1;
    int den = i;
    //greedily simplify the new terms
    if (num % den == 0) {
      numerator *= num/den;
    }
    else {
      numerator *= num;
      denominator *= den;
    }
    //greedily simplify the totals
    if (numerator % denominator == 0) {
      numerator = numerator / denominator;
      denominator = 1;
    }
  }
  return numerator / denominator;
}

bool mss_iter(mss_iter_t *restrict it) {
  if (it->fin) return false;

  const size_t n = it->n, tot = it->tot;
  size_t *vec = it->vec, *scratch = it->scratch;

  if (it->lvl == 0) {
    for (size_t i = 0; i < n - 1; ++i) {
      scratch[i] = (i == 0) ? tot : scratch[i-1] - vec[i-1];
      vec[i] = 0;
    }
    vec[n-1] = (n == 1) ? tot : scratch[n-2] - vec[n-2];
    it->lvl = n;
    return true;
  }

  for (int k = n - 2; k >= 0; --k) {
    size_t hi = (k == 0) ? tot : scratch[k-1] - vec[k-1];
    if (vec[k] < hi) {
      ++vec[k];

      for (size_t i = k + 1; i < n - 1; ++i) {
        scratch[i] = scratch[i-1] - vec[i-1];
        vec[i]     = 0;
      }
      vec[n-1] = scratch[n-2] - vec[n-2];
      return true;
    }
  }

  it->fin = true;
  return false;
}

static uint64_t m_for(uint64_t n) {
  uint64_t x = (n+1)/2;
  if (x & 1) return x;
  return (n+3)/2;
}

void print_vec(const size_t *v, size_t len) {
  putchar('[');
  for (size_t i = 0; i < len; ++i)
    printf("%zu%s", v[i], ((i+1==len) ? "" : ","));
  puts("]");
}

void create_exps(size_t *ms, size_t len, uint64_t *dst) {
  size_t idx = 0;

  for (size_t exp = 0; exp < len; ++exp) {
    for (size_t k = 0; k < ms[exp]; ++k)
      dst[idx++] = exp;
  }

  dst[idx] = 0;
}

static inline uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t p_dash) {
  uint128_t t = (uint128_t)a * b;
  uint64_t m = (uint64_t)t * p_dash;
  uint128_t u = t + (uint128_t)m * p;
  uint64_t res = u >> 64;
  if (res >= p) res -= p;
  return res;
}

#define PRIME_BITS 61
static size_t primes_needed(uint64_t n) {
  // theorem #4
  double log2En = ((n-2.0)*(n+1.0)/2.0)*log2(n)
                - (n*n)/2.0 * M_LOG2E
                + (n+1.0)/2.0
                + 0.5*log2(M_PI)
                + (11.0/12.0) * M_LOG2E;
  size_t bits = (size_t)ceil(log2En);
  return (bits+PRIME_BITS-1)/PRIME_BITS;
}

typedef struct {
  uint64_t n, m, p, p_dash, r, r2, *jk_prod_M, *jk_prod, *nat_M, *jk_sums_M;
  size_t *binoms;
} prim_ctx_t;

static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  return j*m + k;
}

prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t m, uint64_t p, uint64_t w) {
  prim_ctx_t *ctx = malloc(sizeof(prim_ctx_t));
  ctx->n = n;
  ctx->m = m;
  ctx->p = p;
  ctx->p_dash = (uint64_t)(0 - inv64_u64(p));
  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;

  // TODO: move all of the locals into M-space for neatness. wouldn't really
  // affect total run-time cause it's all per-prime only.
  uint64_t ws[m];
  for (size_t i = 0; i < m; ++i)
    ws[i] = pow_mod_u64(w, i, p);
  uint64_t ws_inv[m];
  for (size_t i = 0; i < m; ++i)
    ws_inv[i] = inv_mod_u64(ws[i], p);
  uint64_t jk_pairs[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_pairs[jk_pos(j, k, m)] = mul_mod_u64(ws[j], ws_inv[k], p);
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

void prim_ctx_free(prim_ctx_t *ctx) {
  free(ctx->binoms);
  free(ctx->jk_sums_M);
  free(ctx->nat_M);
  free(ctx->jk_prod_M);
  free(ctx->jk_prod);
  free(ctx);
}

void composition_unrank(size_t *C, size_t rank, size_t m, size_t tot, size_t *vec) {
  for (size_t i=0; i<m-1; ++i) {
    size_t v = 0;
    for (;;) {
      size_t cnt = C[binom_pos(tot-v+m-i-2, m-i-2, m)];
      if (rank < cnt) break;
      rank -= cnt;
      ++v;
    }
    vec[i] = v;
    tot -= v;
  }
  vec[m-1] = tot;
}

uint64_t multinomial_mod_p(prim_ctx_t *ctx, const size_t *ms, size_t len) {
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

  return mont_mul(acc, 1, p, p_dash);
}

uint64_t det_mod_p(uint64_t *A, size_t dim, prim_ctx_t *ctx) {
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

uint64_t f_fst_term(uint64_t *exps, prim_ctx_t *ctx) {
  uint64_t acc = ctx->r;
  for (size_t j = 0; j < ctx->n; ++j) {
    for (size_t k = j + 1; k < ctx->n; ++k) {
      uint64_t t = ctx->jk_sums_M[jk_pos(exps[j], exps[k], ctx->m)];
      acc = mont_mul(acc, t, ctx->p, ctx->p_dash);
    }
  }
  return mont_mul(acc, 1, ctx->p, ctx->p_dash);
}

static void build_drop_mat(uint64_t *A, size_t dim, uint64_t *exps, prim_ctx_t *ctx) {
  const size_t r = dim+1;
  const uint64_t p = ctx->p, m = ctx->m;

  for (size_t j = 0; j < dim; ++j) {
    uint64_t acc = 0;

    for (size_t k = 0; k < r; ++k) if (j != k) {
      const size_t pos = jk_pos(exps[j], exps[k], m);
      uint64_t t = ctx->jk_prod_M[pos];

      acc = add_mod_u64(acc, t, p);

      if (k < dim) {
        A[j*dim + k] = (t == 0) ? 0 : p - t;
      }
    }

    A[j*dim + j] = acc;
  }
}

uint64_t f_snd_trm(uint64_t *vec, uint64_t *exps, prim_ctx_t *ctx) {
  size_t dim = ctx->n-1;
  uint64_t A[dim*dim];

  build_drop_mat(A, dim, exps, ctx);
  return mont_mul(det_mod_p(A, dim, ctx), 1, ctx->p, ctx->p_dash);
}

static inline uint64_t sub_mod_u64(uint64_t x, uint64_t y, uint64_t p) {
  return (x >= y) ? x - y : x + p - y;
}

// TODO: move back into montgomery domain?
uint64_t f_snd_trm_fast(uint64_t *vec, uint64_t *exps, prim_ctx_t *ctx) {
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
  return mul_mod_u64(prod_int, det_q, p);
}

uint64_t f_snd_trm_cmp(uint64_t *vec, uint64_t *exps, prim_ctx_t *ctx) {
  uint64_t slow = f_snd_trm(vec, exps, ctx);
  uint64_t fast = f_snd_trm_fast(vec, exps, ctx);

  if (fast != slow) {
    print_vec(vec, ctx->m);
    print_vec(exps, ctx->n);
    printf("got %"PRIu64", wanted %"PRIu64" (mod %"PRIu64")\n\n", fast, slow, ctx->p);
    abort();
  }

  return slow;
}

uint64_t f(uint64_t *vec, uint64_t *exps, prim_ctx_t *ctx) {
  return mul_mod_u64(f_fst_term(exps, ctx), f_snd_trm_fast(vec, exps, ctx), ctx->p);
}

typedef struct {
  _Atomic size_t *done;
  size_t tot;
  bool quit;
  pthread_cond_t cv;
  pthread_mutex_t mu;
  struct timespec start;
} progress_t;

#if __APPLE__
#  define _CLOCK CLOCK_REALTIME
#else
#  define _CLOCK CLOCK_MONOTONIC
#endif

void *progress(void *_ud) {
  progress_t *ud = _ud;
  size_t tot = ud->tot;
  _Atomic size_t *done = ud->done;

  pthread_mutex_lock(&ud->mu);
  while (!ud->quit) {
    size_t d = atomic_load_explicit(done, memory_order_relaxed);
    double pct = 100.0 * d / tot;
    struct timespec now;
    clock_gettime(_CLOCK, &now);
    double elapsed = (now.tv_sec - ud->start.tv_sec) + (now.tv_nsec - ud->start.tv_nsec)*1e-9;
    double eta = (d && d < tot) ? elapsed * (tot - d) / d : 0.0;
    int eh = elapsed / 3600, es = (int)elapsed % 60, em = ((int)elapsed / 60) % 60;
    int th = (eta / 3600), ts = (int)eta % 60, tm = ((int)eta / 60) % 60;
    fprintf(stderr, "\r%5.2f%% | %02d:%02d:%02d | ETA %02d:%02d:%02d",
            pct, eh, em, es, th, tm, ts);
    if (d >= tot) break;
    now.tv_sec += 1;
    pthread_cond_timedwait(&ud->cv, &ud->mu, &now);
  }
  pthread_mutex_unlock(&ud->mu);
  fprintf(stderr, "\r");
  return NULL;
}

typedef struct { mpz_t X, M; } comb_ctx_t;

static comb_ctx_t *comb_ctx_new() {
  comb_ctx_t *ctx = malloc(sizeof(comb_ctx_t));
  mpz_inits(ctx->X, ctx->M, NULL);
  mpz_set_ui(ctx->X, 0);
  mpz_set_ui(ctx->M, 1);
  return ctx;
}

static void comb_ctx_free(comb_ctx_t *ctx) {
  mpz_clears(ctx->X, ctx->M, NULL);
  free(ctx);
}

static bool comb_ctx_add(comb_ctx_t *ctx, uint64_t res, uint64_t p) {
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

typedef enum {
  MODE_NONE = 0,
  MODE_PROCESS = (1 << 0),
  MODE_COMBINE = (1 << 1),
  MODE_BOTH = MODE_PROCESS|MODE_COMBINE,
  MODE_LAST = MODE_BOTH+1,
} prog_mode_t;

static uint64_t parse_uint(const char *s) {
  char *e;
  uint64_t n = strtoull(s, &e, 10);
  if (s == e || *e != 0) {
    fprintf(stderr, "invalid uint: %s\n", s);
    abort();
  }
  return n;
}

typedef struct source_St {
  int  (*next)(struct source_St *, uint64_t *res, uint64_t *p);
  void (*destroy)(struct source_St *);
  void *state;
} source_t;

static int stdin_next(source_t *, uint64_t *res, uint64_t *p) {
  return scanf("%"SCNu64" %% %"SCNu64, res, p) == 2;
}

static void stdin_destroy(source_t *src) {
  free(src);
}

source_t *source_stdin_new(void) {
  source_t *src = malloc(sizeof *src);
  *src = (source_t){ .next = stdin_next, .destroy = stdin_destroy, .state = NULL };
  return src;
}

typedef struct {
  uint64_t n, m, *ps;
  size_t idx, np;
} proc_state_t;

static uint64_t residue_for_prime(uint64_t n, uint64_t m, uint64_t p) {
  uint64_t w = mth_root_mod_p(p, m);
  prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

  const size_t mss_siz = mss_iter_size(m, n);

  _Atomic size_t done = 0;
  progress_t st = {
    .done = &done, .tot = mss_siz, .quit = false, .mu = PTHREAD_MUTEX_INITIALIZER,
#if __APPLE__
    .cv = PTHREAD_COND_INITIALIZER
#endif
  };
  clock_gettime(_CLOCK, &st.start);
#if !__APPLE__
  pthread_condattr_t ca;
  pthread_condattr_init(&ca);
  pthread_condattr_setclock(&ca, CLOCK_MONOTONIC);
  pthread_cond_init(&st.cv, &ca);
  pthread_condattr_destroy(&ca);
#endif
  pthread_t prog;
  pthread_create(&prog, NULL, progress, &st);

  _Atomic size_t next_rank = 0;

  uint64_t acc = 0;
  #pragma omp parallel
  {
    uint64_t exps[n], l_acc = 0;
    size_t vec[m], scratch[m];
    mss_iter_t it;

    #define CHUNK 1024
    size_t base;
    while ((base = atomic_fetch_add_explicit(&next_rank, CHUNK, memory_order_relaxed)) < mss_siz) {
      size_t lim = base+CHUNK;
      if (lim > mss_siz) lim = mss_siz;

      composition_unrank(ctx->binoms, base, m, n-1, vec);
      mss_iter_init_at(&it, m, n-1, vec, scratch);

      for (size_t r = base; r < lim; ++r) {
        create_exps(vec, m, exps);
        uint64_t coeff = multinomial_mod_p(ctx, vec, m);
        uint64_t f_n = mul_mod_u64(coeff, f(vec, exps, ctx), p);
        l_acc = add_mod_u64(l_acc, f_n, p);

        //if (r + 1 < lim)
        //  VERIFY(mss_iter(&it));
        mss_iter(&it);
      }

      atomic_fetch_add_explicit(&done, lim-base, memory_order_relaxed);
    }

    #pragma omp critical
    acc = add_mod_u64(acc, l_acc, p);
  }

  pthread_mutex_lock(&st.mu);
  st.quit = true;
  pthread_cond_signal(&st.cv);
  pthread_mutex_unlock(&st.mu);
  pthread_join(prog, NULL);
  pthread_cond_destroy(&st.cv);
  pthread_mutex_destroy(&st.mu);

  prim_ctx_free(ctx);
  uint64_t denom = pow_mod_u64(pow_mod_u64(m % p, n - 1, p), p - 2, p);
  return mul_mod_u64(acc, denom, p);
}

static int proc_next(source_t *self, uint64_t *res, uint64_t *p) {
  proc_state_t *st = self->state;
  if (st->idx == st->np) return 0;

  *p = st->ps[st->idx++];
  *res = residue_for_prime(st->n, st->m, *p);
  printf("%" PRIu64 " %% %" PRIu64 "\n", *res, *p);
  return 1;
}

static void proc_destroy(source_t *self) {
  proc_state_t *st = self->state;
  free(st->ps);
  free(st);
  free(self);
}

#define P_STRIDE (1ULL << 10)

static uint64_t *build_prime_list(uint64_t n, uint64_t m, uint64_t m_id, size_t *out_np) {
  size_t np = primes_needed(n);
  uint64_t *ps = malloc(np * sizeof(*ps));

  uint64_t stride = P_STRIDE*m;
  uint64_t p_base = 1ULL + m*(((1ULL << (PRIME_BITS-1)) + m - 2) / m) + m_id*stride;
  for (size_t i = 0; i < np; ++i) {
    ps[i] = prime_congruent_1_mod_m(p_base, m);
    p_base = ps[i] + stride;
  }
  *out_np = np;
  return ps;
}

source_t *source_process_new(uint64_t n, uint64_t m_id) {
  uint64_t m = m_for(n);
  size_t np;
  uint64_t *ps = build_prime_list(n, m, m_id, &np);

  proc_state_t *st = malloc(sizeof(*st));
  *st = (proc_state_t){ .n = n, .m = m, .idx = 0, .np = np, .ps = ps };

  source_t *src = malloc(sizeof *src);
  *src = (source_t){ .next = proc_next, .destroy = proc_destroy, .state = st };
  return src;
}

int main (int argc, char **argv) {
  uint64_t n = 13, m_id = 0;
  prog_mode_t mode = MODE_NONE;

  for (;;) {
    int c = getopt(argc, argv, "m:pc");
    if (c == -1) break;

    switch (c) {
      case 'm': m_id = parse_uint(optarg); break;
      case 'p': mode |= MODE_PROCESS; break;
      case 'c': mode |= MODE_COMBINE; break;
    }
  }
  assert(m_id < P_STRIDE);
  assert(mode < MODE_LAST);
  if (mode == MODE_NONE) mode = MODE_BOTH;

  if (argc > optind)
    n = parse_uint(argv[optind]);

  source_t *src = (mode & MODE_PROCESS) ? source_process_new(n, m_id) : source_stdin_new();
  comb_ctx_t *crt = (mode & MODE_COMBINE) ? comb_ctx_new() : NULL;

  bool converged = false;
  size_t i = 0;
  uint64_t res, p;
  while (src->next(src, &res, &p) > 0) {
    if (mode & MODE_COMBINE) {
      converged = comb_ctx_add(crt, res, p);

      if (i > 0) {
        gmp_printf("e(%"PRIu64") %s %Zd\n  after %zu primes, mod %Zd\n",
                   n, converged ? "=" : ">=", crt->X, i+1, crt->M);
        if (converged && mode & MODE_PROCESS) break;
      }
      ++i;
    }
  }
  src->destroy(src);

  if (mode & MODE_COMBINE) {
    if (!converged)
      gmp_printf("(INCOMPLETE) e_n = %Zd (mod %Zd)\n", crt->X, crt->M);
    comb_ctx_free(crt);
  }

  return 0;
}
