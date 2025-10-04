#pragma once

#include "debug.h"

#include <stddef.h>
#include <stdint.h>

typedef unsigned __int128 uint128_t;

// Limiting prime size to 63 bits stops this overflowing
static inline uint64_t add_mod_u64(uint64_t x, uint64_t y, uint64_t p) {
  x += y;
  int64_t maybe = x - p;
  return maybe < 0 ? x : (uint64_t)maybe;
}

static inline uint64_t mul_mod_u64(uint64_t x, uint64_t y, uint64_t p) {
  assert(x < p);
  assert(y < p);
  return (uint64_t)((uint128_t)x * y % p);
}

static inline uint64_t pow_mod_u64(uint64_t b, uint64_t e, uint64_t p) {
  assert(b < p);
  assert(e < p);
  uint64_t r = 1;
  while (e) {
    if (e & 1)
      r = mul_mod_u64(r, b, p);
    b = mul_mod_u64(b, b, p);
    e >>= 1;
  }
  return r;
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

// montgomery multiply
// static inline uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t p_dash) {
//   uint128_t t = (uint128_t)a * b;
//   uint64_t m = (uint64_t)t * p_dash;
//   uint128_t u = t + (uint128_t)m * p;
//   uint64_t res = u >> 64;
//   int64_t maybe = res - p;
//   return maybe < 0 ? res : (uint64_t)maybe;
// }

static inline uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t p_dash) {
  uint64_t  result;
  uint64_t scratch;
    asm (
        ".intel_syntax\n"
        "mul     %[b]\n"
        "mov     %[working], %%rdx\n"
        "imul    %%rax, %[p_dash]\n"
        "mul     %[p]\n"
        "add     %%rax, -1\n"
        "adc     %%rdx, %[working]\n"
        "mov     %%rax, %%rdx\n"
        "sub     %%rax, %[p]\n"
        "cmovs   %%rax, %%rdx\n"
        ".att_syntax\n"
        : "=a"(result)             // output in rax
        : [a]"0"(a), [b]"r"(b), [p]"r"(p), [p_dash]"r"(p_dash), [working]"r"(scratch)
        : "rdx", "rdi"
    );

    return result;
}

static inline uint64_t sub_mod_u64(uint64_t x, uint64_t y, uint64_t p) {
  return (x >= y) ? x - y : x + p - y;
}

// Pass `r` (montomery 1) into acc for regular power. If you're just going to
// multiply your power into another number you can put that into acc instead to
// save a multiply
static inline uint64_t mont_pow(uint64_t b, uint64_t e, uint64_t acc, uint64_t p,
                         uint64_t p_dash) {
  // Compute by repeated squaring
  while (e) {
    if (e & 1)
      acc = mont_mul(acc, b, p, p_dash);
    b = mont_mul(b, b, p, p_dash);
    e >>= 1;
  }
  return acc;
}

static inline uint64_t extended_euclidean(uint64_t a, uint64_t b) {
  uint64_t r0 = a;
  uint64_t r1 = b;
  uint64_t s0 = 1;
  uint64_t s1 = 0;
  uint64_t spare;
  size_t n = 0;
  while (r1) {
    // compiler should optimize these two into a single instruction
    uint64_t q = r0 / r1;
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

static inline uint64_t mont_inv_act(uint64_t x, uint64_t r3, uint64_t p,
                             uint64_t p_dash) {
  uint64_t inv = extended_euclidean(x, p);
  // inv gives us a value when multiplied gives 1, for a number that when mont
  // mulled gives 1 we need to times by r. For a number that when mont mulled
  // gives r we need to times by r2 timesing by r2 is the same as mont mulling
  // by r3
  return mont_mul(r3, inv, p, p_dash);
}

#else
#define mont_inv(w, x, y, z) mont_inv_act((w), (y), (z))

static inline uint64_t mont_inv_act(uint64_t x, uint64_t p, uint64_t p_dash) {
  return mont_pow(x, p - 3, x, p, p_dash);
}

#endif

// Does a1 * b1 - a2 * b2
static inline uint64_t mont_mul_sub(uint64_t a1, uint64_t b1, uint64_t a2, uint64_t b2,
                             uint64_t p, uint64_t p_dash) {
  uint128_t t1 = (uint128_t)a1 * b1;
  uint128_t t2 = (uint128_t)a2 * b2;
  uint128_t t = t1 + ((uint128_t)p << 64) - t2;
  uint64_t m = (uint64_t)t * p_dash;
  uint64_t u = (t + (uint128_t)m * p) >> 64;
  int64_t maybe = u - p;
  return maybe < 0 ? u : (uint64_t)maybe;
}
