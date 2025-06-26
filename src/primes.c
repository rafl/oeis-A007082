#include "oeis.h"
#include "debug.h"
#include "primes.h"
#include "maths.h"

#include <stdlib.h>
#include <math.h>
#include <gmp.h>

#define PRIME_BITS 61

static int factor_u64(uint64_t m, uint64_t *pf, DEBUG_ARG size_t pfs, size_t *pcnt) {
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

uint64_t mth_root_mod_p(uint64_t p, uint64_t m) {
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

uint64_t *build_prime_list(uint64_t n, uint64_t m, uint64_t m_id, size_t stride, size_t *out_np) {
  size_t np = primes_needed(n);
  uint64_t *ps = malloc(np * sizeof(*ps));

  uint64_t stride_m = stride*m;
  uint64_t p_base = 1ULL + m*(((1ULL << (PRIME_BITS-1)) + m - 2) / m) + m_id*stride_m;
  for (size_t i = 0; i < np; ++i) {
    ps[i] = prime_congruent_1_mod_m(p_base, m);
    p_base = ps[i] + stride_m;
  }
  *out_np = np;
  return ps;
}
