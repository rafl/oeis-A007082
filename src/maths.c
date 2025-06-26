#include "oeis.h"
#include "maths.h"

uint64_t add_mod_u64(uint64_t x, uint64_t y, uint64_t p) {
  x += y;
  if (x >= p) x -= p;
  return x;
}

uint64_t mul_mod_u64(uint64_t x, uint64_t y, uint64_t p) {
  assert(x < p);
  assert(y < p);
  return (uint64_t)((uint128_t)x * y % p);
}

uint64_t pow_mod_u64(uint64_t b, uint64_t e, uint64_t p) {
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

uint64_t inv_mod_u64(uint64_t x, uint64_t p) {
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

uint64_t inv64_u64(uint64_t p) {
  uint64_t x = 1;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  return x;
}

int factor_u64(uint64_t m, uint64_t *pf, DEBUG_ARG size_t pfs, size_t *pcnt) {
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
