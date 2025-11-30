#pragma once

#include "maths.h"
#include "mss.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Check if CUDA is available at runtime
bool gpu_available(void);

// Get number of available GPUs
int gpu_device_count(void);

// Opaque structures
typedef struct gpu_context_t gpu_context_t;
typedef struct vec_batch_t vec_batch_t;

typedef void (*batch_cb_t)(void *);

// === GPU Context API (shared constant data) ===

// Create GPU context with constant lookup tables (shared across batches)
gpu_context_t *gpu_context_new(
    uint64_t n, uint64_t n_args, uint64_t m, fld_t p, fld_t p_dash,
    fld_t r, fld_t r3, const fld_t *jk_prod_M, const fld_t *nat_M,
    const fld_t *nat_inv_M, const fld_t *ws_M,
    const fld_t *jk_sums_pow_lower_M, const fld_t *jk_sums_pow_upper_M,
    const fld_t *rs, const fld_t *fact_M, const fld_t *fact_inv_M,
    size_t m_half, size_t n_rs, bool is_jack_mode);

// Free GPU context
void gpu_context_free(gpu_context_t *ctx);

// === Vector Batch API ===

// Create a vector batch that uses a shared GPU context
vec_batch_t *vec_batch_new(gpu_context_t *ctx, size_t max_vecs);

// Add a coefficient vector to the batch
size_t vec_batch_add(vec_batch_t *batch, const mss_el_t *vec);

// Add multiple coefficient vectors to the batch in bulk (more efficient)
void vec_batch_add_bulk(vec_batch_t *batch, const mss_el_t *vecs, size_t count);

// Launch async GPU compute (returns immediately)
void vec_batch_compute_async(vec_batch_t *batch, batch_cb_t done, void *ud);

// Get result for a vector
uint64_t vec_batch_get(const vec_batch_t *batch, size_t idx);

// Clear batch for reuse
void vec_batch_clear(vec_batch_t *batch);

// Free batch
void vec_batch_free(vec_batch_t *batch);

#ifdef __cplusplus
}
#endif
