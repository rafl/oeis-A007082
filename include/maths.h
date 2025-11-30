#pragma once

#include "debug.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#define PRIME_BITS 31

// mostly for debugging
#ifndef FLD_TYPE_BITS
#define FLD_TYPE_BITS PRIME_BITS
#endif

_Static_assert(PRIME_BITS < 64);
_Static_assert(PRIME_BITS <= FLD_TYPE_BITS);

#if FLD_TYPE_BITS > 31
typedef uint64_t fld_t;
typedef int64_t sfld_t;
typedef unsigned __int128 dfld_t;
#define FLD_BITS 64
#define FLD_FMT PRIu64
#elif FLD_TYPE_BITS > 15
typedef uint32_t fld_t;
typedef int32_t sfld_t;
typedef uint64_t dfld_t;
#define FLD_BITS 32
#define FLD_FMT PRIu32
#elif FLD_TYPE_BITS > 7
typedef uint16_t fld_t;
typedef int16_t sfld_t;
typedef uint32_t dfld_t;
#define FLD_BITS 16
#define FLD_FMT PRIu16
#else
typedef uint8_t fld_t;
typedef int8_t sfld_t;
typedef uint16_t dfld_t;
#define FLD_BITS 8
#define FLD_FMT PRIu8
#endif

typedef unsigned __int128 uint128_t;

// Limiting prime size to 63 bits stops this overflowing
inline fld_t add_mod_u64(fld_t x, fld_t y, fld_t p) {
  x += y;
  sfld_t maybe = x - p;
  return maybe < 0 ? x : (fld_t)maybe;
}

inline fld_t mul_mod_u64(fld_t x, fld_t y, fld_t p) {
  assert(x < p);
  assert(y < p);
  return (fld_t)((dfld_t)x * y % p);
}

inline fld_t pow_mod_u64(fld_t b, uint64_t e, fld_t p) {
  assert(b < p);
  assert(e < p);
  fld_t r = 1;
  while (e) {
    if (e & 1)
      r = mul_mod_u64(r, b, p);
    b = mul_mod_u64(b, b, p);
    e >>= 1;
  }
  return r;
}

inline fld_t inv64_u64(fld_t p) {
  fld_t x = 1;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  return x;
}

// montgomery multiply
inline fld_t mont_mul(fld_t a, fld_t b, fld_t p, fld_t p_dash) {
  dfld_t t = (dfld_t)a * b;
  fld_t m = (fld_t)t * p_dash;
  dfld_t u = t + (dfld_t)m * p;
  fld_t res = u >> FLD_BITS;
  sfld_t maybe = res - p;
  return maybe < 0 ? res : (fld_t)maybe;
}

inline fld_t sub_mod_u64(fld_t x, fld_t y, fld_t p) {
  return (x >= y) ? x - y : x + p - y;
}

// Pass `r` (montomery 1) into acc for regular power. If you're just going to
// multiply your power into another number you can put that into acc instead to
// save a multiply
inline fld_t mont_pow(fld_t b, uint64_t e, fld_t acc, fld_t p,
                         fld_t p_dash) {
  // Compute by repeated squaring
  while (e) {
    if (e & 1)
      acc = mont_mul(acc, b, p, p_dash);
    b = mont_mul(b, b, p, p_dash);
    e >>= 1;
  }
  return acc;
}

inline fld_t extended_euclidean(fld_t a, fld_t b) {
  fld_t r0 = a;
  fld_t r1 = b;
  fld_t s0 = 1;
  fld_t s1 = 0;
  fld_t spare;
  size_t n = 0;
  while (r1) {
    // compiler should optimize these two into a single instruction
    fld_t q = r0 / r1;
    spare = r0 % r1;
    r0 = r1;
    r1 = spare;
    spare = s0 + q * s1;
    s0 = s1;
    s1 = spare;
    ++n;
  }
  // gcd = r0
  if (n % 2)
    s0 = b - s0;
  return s0;
}

#if !SLOW_DIVISION
#define mont_inv(w, x, y, z) mont_inv_act((w), (x), (y), (z))

inline fld_t mont_inv_act(fld_t x, fld_t r3, fld_t p,
                             fld_t p_dash) {
  fld_t inv = extended_euclidean(x, p);
  // inv gives us a value when multiplied gives 1, for a number that when mont
  // mulled gives 1 we need to times by r. For a number that when mont mulled
  // gives r we need to times by r2 timesing by r2 is the same as mont mulling
  // by r3
  return mont_mul(r3, inv, p, p_dash);
}

#else
#define mont_inv(w, x, y, z) mont_inv_act((w), (y), (z))

inline fld_t mont_inv_act(fld_t x, fld_t p, fld_t p_dash) {
  return mont_pow(x, p - 3, x, p, p_dash);
}

#endif

// Does a1 * b1 - a2 * b2
inline fld_t mont_mul_sub(fld_t a1, fld_t b1, fld_t a2, fld_t b2,
                             fld_t p, fld_t p_dash) {
  dfld_t t1 = (dfld_t)a1 * b1;
  dfld_t t2 = (dfld_t)a2 * b2;
  dfld_t t = t1 + ((dfld_t)p << FLD_BITS) - t2;
  fld_t m = (fld_t)t * p_dash;
  fld_t u = (t + (dfld_t)m * p) >> FLD_BITS;
  if (u >= p) u -= p;
  if (u >= p) u -= p;
  assert(u < p);
  return u;
}

inline uint64_t m_for(uint64_t n) { return 2 * ((n + 1) / 4) + 1; }
