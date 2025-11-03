#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Check if CUDA is available at runtime
bool gpu_det_available(void);

// Opaque batch structure
typedef struct det_batch_t det_batch_t;
typedef struct vec_batch_t vec_batch_t;

// === Vector-based API (builds matrices on GPU, much faster) ===

// Create a vector batch for processing coefficient vectors on GPU
vec_batch_t *vec_batch_new(size_t max_vecs, uint64_t m, uint64_t p,
                            uint64_t p_dash, uint64_t r, uint64_t r3,
                            const uint64_t *jk_prod_M, const uint64_t *nat_M,
                            const uint64_t *nat_inv_M, bool is_jack_mode);

// Add a coefficient vector to the batch
size_t vec_batch_add(vec_batch_t *batch, const uint64_t *vec);

// Compute all f_snd_trm results on GPU (builds matrices + computes dets)
void vec_batch_compute(vec_batch_t *batch);

// Get result for a vector
uint64_t vec_batch_get(const vec_batch_t *batch, size_t idx);

// Clear batch for reuse
void vec_batch_clear(vec_batch_t *batch);

// Free batch
void vec_batch_free(vec_batch_t *batch);

// === OLD Matrix-based API (for reference/fallback) ===

// Create a new determinant batch
// max_matrices: maximum number of matrices in the batch
// max_dim: maximum dimension of any matrix (all matrices padded to this)
// p, p_dash, r, r3: Montgomery arithmetic parameters
det_batch_t *det_batch_new(size_t max_matrices, size_t max_dim, uint64_t p,
                           uint64_t p_dash, uint64_t r, uint64_t r3);

// Add a matrix to the batch
// matrix: flattened matrix data (dim x dim, row-major)
// dim: actual dimension of this matrix
// prod_M: additional scalar to multiply result by (for f_snd_trm pattern)
// Returns: index of this matrix in the batch
size_t det_batch_add(det_batch_t *batch, const uint64_t *matrix, size_t dim,
                     uint64_t prod_M);

// Compute all determinants in the batch on GPU
// After this, results can be retrieved with det_batch_get
void det_batch_compute(det_batch_t *batch);

// Get the result for a specific matrix
// idx: index returned from det_batch_add
// Returns: mont_mul(prod_M, det_mod_p(A, dim), p, p_dash)
uint64_t det_batch_get(const det_batch_t *batch, size_t idx);

// Clear the batch for reuse (keeps allocated memory)
void det_batch_clear(det_batch_t *batch);

// Free the batch
void det_batch_free(det_batch_t *batch);

#ifdef __cplusplus
}
#endif
