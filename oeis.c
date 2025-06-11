#define DEBUG 1
#if !DEBUG
#  define NDEBUG
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>
#include <string.h>
#include <gmp.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>

typedef unsigned __int128 uint128_t;

static inline uint64_t mul_mod_u64(uint64_t x, uint64_t y, uint64_t p) {
  assert(x < p);
  assert(y < p);
  return (uint64_t)((uint128_t)x * y % p);
}

static uint64_t pow_mod_u64(uint64_t b, uint64_t e, uint64_t p) {
  assert(b < p);
  assert(e < p);
  uint64_t r = 1;
  while (e) {
    if (e & 1) r = mul_mod_u64(r, b, p);
    b = mul_mod_u64(b, b, p);
    e >>= 1;
  }
  return r;
}

static inline uint64_t inv_mod_u64(uint64_t x, uint64_t p) {
  assert(p < (1ULL<<63));
  assert(x < p);
  int64_t t = 0, new_t = 1, r = (int64_t)p, new_r = (int64_t)x;

  while (new_r) {
    int64_t tmp, q = r / new_r;
    tmp = r - q * new_r; r = new_r; new_r = tmp;
    tmp = t - q * new_t; t = new_t; new_t = tmp;
  }

  assert(r == 1);
  if (t < 0) t += p;
  return (uint64_t)t;
}

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

static int factor_u64(uint64_t m, uint64_t *pf, size_t pfs, size_t *pcnt) {
  size_t k = 0;
  for (uint64_t d = 2; d * d <= m; ++d) {
    if (m % d == 0) {
      assert(k < pfs);
      pf[k++] = d;
      while (m % d == 0) m /= d;
    }
  }
  assert(k < pfs);
  if (m > 1) pf[k++] = m;
  assert(k < pfs);
  *pcnt = k;
  return 0;
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

  dst[idx] = 1;
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
  uint64_t n, m, p, p_dash, r, r2, *ws, *ws_inv, *jk_prod_M, *nat_M, *jk_sums_M;
} prim_ctx_t;

static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  return j*m + k;
}

static inline uint64_t inv64_u64(uint64_t p) {
  uint64_t x = 1;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  return x;
}

prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t m, uint64_t p, uint64_t w) {
  prim_ctx_t *ctx = malloc(sizeof(prim_ctx_t));
  ctx->n = n;
  ctx->m = m;
  ctx->p = p;
  ctx->p_dash = (uint64_t)(0 - inv64_u64(p));
  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;
  ctx->ws = malloc(m*sizeof(uint64_t));
  for (size_t i = 0; i < m; ++i) {
    ctx->ws[i] = pow_mod_u64(w, i, p);
  }
  ctx->ws_inv = malloc(m*sizeof(uint64_t));
  for (size_t i = 0; i < m; ++i) {
    ctx->ws_inv[i] = inv_mod_u64(ctx->ws[i], p);
  }
  uint64_t jk_pairs[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_pairs[jk_pos(j, k, m)] = mul_mod_u64(ctx->ws[j], ctx->ws_inv[k], p);
  }
  uint64_t jk_sums[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_sums[jk_pos(j, k, m)] =
        ((uint128_t)jk_pairs[jk_pos(j, k, m)] + jk_pairs[jk_pos(k, j, m)]) % p;
  }
  ctx->jk_prod_M = malloc(m*m*sizeof(uint64_t));
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      size_t pos = jk_pos(j, k, m);
      uint64_t t = mul_mod_u64(jk_pairs[pos], inv_mod_u64(jk_sums[pos], p), p);
      ctx->jk_prod_M[pos] = mont_mul(t, ctx->r2, p, ctx->p_dash);
    }
  }
  ctx->nat_M = malloc((n+1)*sizeof(uint64_t));
  for (size_t i = 1; i <= n; ++i)
    ctx->nat_M[i] = mont_mul(i, ctx->r2, p, ctx->p_dash);
  ctx->jk_sums_M = malloc(m*m*sizeof(uint64_t));
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      size_t pos = jk_pos(j,k,m);
      ctx->jk_sums_M[pos] = mont_mul(jk_sums[pos], ctx->r2, p, ctx->p_dash);
    }
  }

  return ctx;
}

void prim_ctx_free(prim_ctx_t *ctx) {
  free(ctx->jk_sums_M);
  free(ctx->nat_M);
  free(ctx->jk_prod_M);
  free(ctx->ws_inv);
  free(ctx->ws);
  free(ctx);
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

      acc = acc + t;
      if (acc >= p) acc -= p;

      if (k < dim) {
        A[j*dim + k] = (t == 0) ? 0 : p - t;
      }
    }

    A[j*dim + j] = acc;
  }
}

uint64_t f_snd_trm(uint64_t *exps, prim_ctx_t *ctx) {
  size_t dim = ctx->n-1;
  uint64_t A[dim*dim];

  build_drop_mat(A, dim, exps, ctx);
  return mont_mul(det_mod_p(A, dim, ctx), 1, ctx->p, ctx->p_dash);
}

uint64_t f(uint64_t *exps, prim_ctx_t *ctx) {
  return mul_mod_u64(f_fst_term(exps, ctx), f_snd_trm(exps, ctx), ctx->p);
}

int main(int argc, char **argv) {
  uint64_t n = 13;

  if (argc > 1) {
    n = atoll(argv[1]);
  }

  uint64_t m = m_for(n);
  printf("n = %"PRIu64", m = %"PRIu64"\n", n, m);

  size_t np = primes_needed(n);
  printf("np = %zu\n", np);
  uint64_t ps[np];
  uint64_t p_base = 1ULL << (PRIME_BITS-1);
  for (size_t i = 0; i < np; ++i) {
    ps[i] = prime_congruent_1_mod_m(p_base, m);
    p_base = ps[i]+1;
  }

  mpz_t X, M, u, inv, mz, rz;
  mpz_inits(X, M, u, inv, mz, rz, NULL);
  mpz_set_ui(X, 0);
  mpz_set_ui(M, 1);

  #pragma omp parallel for
  for (size_t i = 0; i < np; ++i) {
    uint64_t p = ps[i];
    printf("p = %"PRIu64"\n", p);

    uint64_t w = mth_root_mod_p(p, m);
    prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

    size_t vec[m], scratch[m];
    uint64_t exps[n], acc = 0;
    mss_iter_t it;
    mss_iter_new(&it, m, n-1, vec, scratch);
    while (mss_iter(&it)) {
      create_exps(vec, m, exps);
      uint64_t coeff = multinomial_mod_p(ctx, vec, m);
      uint64_t f_n = mul_mod_u64(coeff, f(exps, ctx), p);
      acc = acc + f_n;
      if (acc >= p) acc -= p;
    }
    uint64_t ret = mul_mod_u64(acc, pow_mod_u64(pow_mod_u64(m % p, n - 1, p), p - 2, p), p);
    printf("%"PRIu64" %% %"PRIu64"\n", ret, p);

    mpz_set_ui(rz, ret);
    mpz_set_ui(mz, p);

    mpz_mod(u, X, mz);
    mpz_sub(u, rz, u);
    mpz_mod(u, u, mz);

    assert(mpz_invert(inv, M, mz) != 0);

    mpz_mul(u, u, inv);
    mpz_mod(u, u, mz);
    mpz_mul(inv, M, u);
    mpz_add(X, X, inv);
    mpz_mul(M, M, mz);

    prim_ctx_free(ctx);
  }

  gmp_printf("e_n = %Zd (mod %Zd)\n", X, M);
  mpz_clears(X, M, u, inv, mz, rz, NULL);

  return 0;
}
