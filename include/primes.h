#pragma once

#include "maths.h"

#include <stddef.h>
#include <stdint.h>

fld_t prime_congruent_1_mod_m(fld_t, uint64_t);
fld_t mth_root_mod_p(fld_t, uint64_t);
fld_t *build_prime_list(uint64_t, uint64_t, uint64_t, size_t, size_t *);
