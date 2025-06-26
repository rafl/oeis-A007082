#include "debug.h"
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

uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t p_dash) {
  uint128_t t = (uint128_t)a * b;
  uint64_t m = (uint64_t)t * p_dash;
  uint128_t u = t + (uint128_t)m * p;
  uint64_t res = u >> 64;
  if (res >= p) res -= p;
  return res;
}

inline uint64_t sub_mod_u64(uint64_t x, uint64_t y, uint64_t p) {
  return (x >= y) ? x - y : x + p - y;
}
