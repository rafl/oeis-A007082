#include "debug.h"
#include "maths.h"

// Limiting prime size to 63 bits stops this overflowing
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

// montgomery multiply
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

uint64_t mont_pow(uint64_t b, uint64_t e, uint64_t r, uint64_t p, uint64_t p_dash) {
  uint64_t acc = r;
  // Compute by repeated squaring
  while (e) {
    if (e & 1)
      acc = mont_mul(acc, b, p, p_dash);
    b = mont_mul(b, b, p, p_dash);
    e >>= 1;
  }
  return acc;
}

uint64_t old_mont_inv(uint64_t x, uint64_t r, uint64_t p, uint64_t p_dash) {
  return mont_pow(x, p-2, r, p, p_dash);
}

typedef struct {
  uint64_t r_savas;
  uint64_t k;
} alm_inv_t;

alm_inv_t alm_mod_inv(uint64_t a, uint64_t p) {
  uint64_t u = p;
  uint64_t v = a;
  uint64_t r_savas = 0;
  uint64_t s = 1;
  uint64_t k = 0;
  while (v > 0)
  {
    if (u % 2 == 0)
    {
      u /=2 ;
      s*=2;
    } else if (v % 2 == 0)
    {
      v /= 2;
      r_savas *= 2;
    } else if (u > v) {
      u = (u - v) / 2;
      r_savas = r_savas+s;
      s *= 2;
    } else if (v >= u) {
      v = (v - u) / 2;
      s = s + r_savas;
      r_savas *= 2;
    }
    else {
      assert(0);
    }
    k++;
  }

  if (r_savas >= p)
  {
    r_savas -= p;
  }

  return (alm_inv_t){p -r_savas, k};
  // n = 64
  // m = 64
  // r = n mod p
  // p_dash -> p inverse mod 2^n
}

uint64_t new_mont_inv(uint64_t x, uint64_t r2, uint64_t p, uint64_t p_dash) {
  // x = a*2^m
  alm_inv_t const alm_inv = alm_mod_inv(x, p);
  uint64_t r_savas = alm_inv.r_savas;
  uint64_t k = alm_inv.k;
  uint64_t const m = 64;
  uint64_t const n = 64;
  if (alm_inv.k == m) // if n <= k <=m
  {
    r_savas = mont_mul(r_savas, r2, p, p_dash);
    k += m;
  }

  r_savas = mont_mul(r_savas, r2, p, p_dash);
  uint64_t two_to_2m_minus_k = 1LU << (2*m - k);
  r_savas = mont_mul(r_savas, two_to_2m_minus_k, p, p_dash);
  return r_savas;
  // n = 64
  // m = 64
  // r = n mod p
  // p_dash -> p inverse mod 2^n
}

uint64_t mont_inv(uint64_t x, uint64_t r, uint64_t r2, uint64_t p, uint64_t p_dash)
{
  uint64_t old_val = old_mont_inv(x, r, p, p_dash);
  uint64_t new_val = new_mont_inv(x, r2, p, p_dash);
  if (old_val != new_val)
  {
   assert(old_val == new_val && "mismatch");
  }
  return old_val;
}


// Does a1 * b1 - a2 * b2
uint64_t mont_mul_sub(uint64_t a1, uint64_t b1, uint64_t a2, uint64_t b2, uint64_t p, uint64_t p_dash) {
  uint128_t t1 = (uint128_t)a1 * b1;
  uint128_t t2 = (uint128_t)a2 * b2;
  uint128_t t = t1 + ((uint128_t)p << 64) - t2;
  uint64_t m = (uint64_t)t * p_dash;
  uint64_t u = (t + (uint128_t)m * p) >> 64;
  return (u >= p) ? u - p : u;
}
