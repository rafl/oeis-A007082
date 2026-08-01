#include "maths.h"

#define SIZE 16

#ifdef __cplusplus
extern "C" {
#endif
void det_mod_p_gpu(uint32_t const * values, uint32_t * results, uint32_t n_matricies,      fld_t p,
                                           fld_t p_dash, fld_t r,
                                           fld_t r3);

#ifdef __cplusplus
}
#endif