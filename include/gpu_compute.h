#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Check if CUDA is available
bool gpu_available(void);

// GPU computation context
typedef struct gpu_ctx_t gpu_ctx_t;

// Create GPU context for a specific prime
gpu_ctx_t *gpu_ctx_new(uint64_t n, uint64_t n_args, uint64_t m, uint64_t p,
                       uint64_t w, bool jack_mode);

// Free GPU context
void gpu_ctx_free(gpu_ctx_t *ctx);

// Process a batch of coefficient vectors on GPU
// Returns the sum of all results modulo p
uint64_t gpu_process_batch(gpu_ctx_t *ctx, const size_t *vecs, size_t n_vecs);

#ifdef __cplusplus
}
#endif
