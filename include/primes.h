#pragma once

#include <stddef.h>
#include <stdint.h>

uint64_t prime_congruent_1_mod_m(uint64_t, uint64_t);
uint64_t mth_root_mod_p(uint64_t, uint64_t);
uint64_t *build_prime_list(uint64_t, uint64_t, uint64_t, size_t, size_t *);
