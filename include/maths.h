#pragma once

#include <stdint.h>
#include <stddef.h>

typedef unsigned __int128 uint128_t;

uint64_t add_mod_u64(uint64_t, uint64_t, uint64_t);
uint64_t mul_mod_u64(uint64_t, uint64_t, uint64_t);
uint64_t pow_mod_u64(uint64_t, uint64_t, uint64_t);
uint64_t sub_mod_u64(uint64_t, uint64_t, uint64_t);
uint64_t inv64_u64(uint64_t);
uint64_t mont_mul(uint64_t, uint64_t, uint64_t, uint64_t);
uint64_t mont_pow(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
uint64_t mont_inv(uint64_t, uint64_t, uint64_t, uint64_t);
uint64_t mont_mul_sub(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

template <typename FieldT>
FieldT field_pow(FieldT b, uint64_t e) {
  FieldT acc = FieldT::One(b.mCoefficients.size());
  while (e) {
    if (e & 1) {
      acc.MultiplyBy(b);
    }
    b.MultiplyBy(b);
    e >>= 1;
  }
  return acc;
}
