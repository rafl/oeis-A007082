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


void create_ws(uint64_t p, uint64_t w, size_t *ms, size_t len, uint64_t *dst) {
  uint64_t pow = 1;
  size_t idx = 0;

  for (size_t exp = 0; exp < len; ++exp) {
    for (size_t k = 0; k < ms[exp]; ++k)
      dst[idx++] = pow;

    pow = mul_mod_u64(pow, w, p);
  }

  dst[idx] = 1;
}

uint64_t multinomial_mod_p(uint64_t p, const size_t *ms, size_t len) {
  uint64_t acc = 1, n = 0;

  for (size_t i = 0; i < len; ++i) {
    for (size_t k = 1; k <= ms[i]; ++k)
      acc = mul_mod_u64(acc, ++n, p);

    uint64_t d = 1;
    for (size_t k = 1; k <= ms[i]; ++k)
      d = mul_mod_u64(d, k, p);

    uint64_t dinv = pow_mod_u64(d, p - 2, p);
    acc = mul_mod_u64(acc, dinv, p);
  }
  return acc;
}

static inline uint64_t inv_ell(uint64_t xj, uint64_t xk,
                               uint64_t inv_j, uint64_t inv_k,
                               uint64_t p) {
  uint64_t ell = mul_mod_u64(inv_j, xk, p) + mul_mod_u64(xj, inv_k, p);
  if (ell >= p) ell -= p;
  return pow_mod_u64(ell, p-2, p);
}

static void build_drop_mat(uint64_t *A, size_t dim, const uint64_t *xs, const uint64_t *inv, uint64_t p) {
  size_t m = dim + 1;

  for (size_t j = 0; j < dim; ++j) {
    uint64_t rowsum = 0;

    for (size_t k = 0; k < m; ++k) if (j != k) {
      uint64_t invell = inv_ell(xs[j], xs[k], inv[j], inv[k], p);
      uint64_t t = mul_mod_u64(xs[j], mul_mod_u64(inv[k], invell, p), p);

      rowsum = rowsum + t < p ? rowsum + t : rowsum + t - p;

      if (k < dim) {
        A[j*dim + k] = p - t;
      }
    }
    A[j*dim + j] = rowsum;
  }
}

uint64_t det_mod_p(uint64_t *A, size_t dim, uint64_t p) {
  uint64_t det = 1;

  for (size_t k = 0; k < dim; ++k) {
    size_t pivot = k;
    while (pivot < dim && A[pivot*dim + k] == 0) ++pivot;
    //assert(pivot != 0);
    if (pivot == dim) return 0;
    if (pivot != k) {
      for (size_t j = 0; j < dim; ++j) {
        uint64_t tmp          = A[k*dim + j];
        A[k*dim + j]          = A[pivot*dim + j];
        A[pivot*dim + j]      = tmp;
      }
      det = p - det;
    }

    uint64_t inv_pivot = inv_mod_u64(A[k*dim + k], p);
    det = mul_mod_u64(det, A[k*dim + k], p);

    for (size_t i = k + 1; i < dim; ++i) {
      uint64_t factor = mul_mod_u64(A[i*dim + k], inv_pivot, p);
      if (factor == 0) continue;

      for (size_t j = k; j < dim; ++j) {
        uint64_t tmp = mul_mod_u64(factor, A[k*dim + j], p);
        uint64_t val = (A[i*dim + j] >= tmp)
                     ? A[i*dim + j] - tmp
                     : A[i*dim + j] + p - tmp;
        A[i*dim + j] = val;
      }
    }
  }

  return det;
}

uint64_t f_fst_term(const uint64_t *ws, const uint64_t *inv, size_t n, uint64_t p) {
  uint64_t acc = 1;
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = j + 1; k < n; ++k) {
      uint64_t t = mul_mod_u64(inv[j], ws[k], p) + mul_mod_u64(ws[j], inv[k], p);
      if (t >= p) t -= p; // faster than % p, i think. sum < 2p is guaranteed

      acc = mul_mod_u64(acc, t, p);
    }
  }
  return acc;
}

uint64_t f_snd_trm(uint64_t p, const uint64_t *xs, const uint64_t *inv, size_t m) {
    size_t dim = m-1;
    uint64_t A[dim*dim];

    build_drop_mat(A, dim, xs, inv, p);
    return det_mod_p(A, dim, p);
}

uint64_t f(const uint64_t *ws, size_t n, uint64_t p) {
  uint64_t inv[n];
  for (size_t i = 0; i < n; ++i)
    inv[i] = inv_mod_u64(ws[i], p);

  return mul_mod_u64(f_fst_term(ws, inv, n, p), f_snd_trm(p, ws, inv, n), p);
}

int main () {
  uint64_t n = 11, m = m_for(n);

  printf("n = %"PRIu64", m = %"PRIu64"\n", n, m);

  uint64_t p = prime_congruent_1_mod_m(1ULL << 60, m);
  printf("p = %"PRIu64" (p %% m = %"PRIu64")\n", p, p % m);

  uint64_t w = mth_root_mod_p(p, m);
  printf("w = %"PRIu64"\n", w);
  printf("w^m mod p = %"PRIu64"\n", pow_mod_u64(w, m, p));
  printf("w^-1 = %"PRIu64"\n", inv_mod_u64(w, p));

  size_t vec[m], scratch[m];
  uint64_t ws[n], acc = 0;
  mss_iter_t it;
  mss_iter_new(&it, m, n-1, vec, scratch);
  while (mss_iter(&it)) {
    create_ws(p, w, vec, m, ws);
    uint64_t coeff = multinomial_mod_p(p, vec, m);
    uint64_t f_n = mul_mod_u64(coeff, f(ws, n, p), p);
    acc = acc + f_n;
    if (acc >= p) acc -= p;
  }
  printf("%lu\n", mul_mod_u64(acc, pow_mod_u64(pow_mod_u64(m % p, n - 1, p), p - 2, p), p));

  return 0;
}
