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

// Pass `r` (montomery 1) into acc for regular power. If you're just going to multiply
// your power into another number you can ass that into acc instead to save a multiply
uint64_t mont_pow(uint64_t b, uint64_t e, uint64_t acc, uint64_t p, uint64_t p_dash) {
  // Compute by repeated squaring
  while (e) {
    if (e & 1)
      acc = mont_mul(acc, b, p, p_dash);
    b = mont_mul(b, b, p, p_dash);
    e >>= 1;
  }
  return acc;
}

// Function to compute the extended Euclidean algorithm
// It returns gcd(a, b), and updates x and y such that: a*x + b*y = gcd(a, b)
void extended_euclidean(uint64_t a, uint64_t b, uint64_t *x, uint64_t *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return;
    }

    uint64_t x1, y1;
    extended_euclidean(b, a % b, &x1, &y1);

    *x = y1;
    *y = x1 - (a / b) * y1;

    return;
}

// Function to compute modular inverse of a modulo p (assuming p is prime)
uint64_t mod_inverse(uint64_t a, uint64_t p) {
    uint64_t x, y;
    extended_euclidean(a, p, &x, &y);

    return (x % p + p) % p;
}

uint64_t mont_inv(uint64_t x, uint64_t r, uint64_t p, uint64_t p_dash) {
  return mont_pow(x, p-3, x, p, p_dash);
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
