#include "gpu_det.h"
#include "debug.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      abort();                                                                 \
    }                                                                          \
  } while (0)

typedef unsigned __int128 uint128_t;

// Helper function for jk_pos on device
__device__ inline size_t d_jk_pos(size_t j, size_t k, uint64_t m) {
  int64_t result = k - j;
  return result >= 0 ? (uint64_t)result : result + m;
}

__device__ inline uint64_t d_add_mod(uint64_t x, uint64_t y, uint64_t p) {
  x += y;
  int64_t maybe = x - p;
  return maybe < 0 ? x : (uint64_t)maybe;
}

// Device-side Montgomery arithmetic helpers
__device__ inline uint64_t d_mont_mul(uint64_t a, uint64_t b, uint64_t p,
                                      uint64_t p_dash) {
  uint128_t t = (uint128_t)a * b;
  uint64_t m = (uint64_t)t * p_dash;
  uint128_t u = t + (uint128_t)m * p;
  uint64_t res = u >> 64;
  int64_t maybe = res - p;
  return maybe < 0 ? res : (uint64_t)maybe;
}

__device__ inline uint64_t d_mont_pow(uint64_t b, uint64_t e, uint64_t acc,
                                      uint64_t p, uint64_t p_dash) {
  while (e) {
    if (e & 1)
      acc = d_mont_mul(acc, b, p, p_dash);
    b = d_mont_mul(b, b, p, p_dash);
    e >>= 1;
  }
  return acc;
}

__device__ inline uint64_t d_extended_euclidean(uint64_t a, uint64_t b) {
  uint64_t r0 = a;
  uint64_t r1 = b;
  uint64_t s0 = 1;
  uint64_t s1 = 0;
  uint64_t spare;
  size_t n = 0;
  while (r1) {
    uint64_t q = r0 / r1;
    spare = r0 % r1;
    r0 = r1;
    r1 = spare;
    spare = s0 + q * s1;
    s0 = s1;
    s1 = spare;
    ++n;
  }
  if (n % 2)
    s0 = b - s0;
  return s0;
}

__device__ inline uint64_t d_mont_inv(uint64_t x, uint64_t r3, uint64_t p,
                                      uint64_t p_dash) {
  uint64_t inv = d_extended_euclidean(x, p);
  return d_mont_mul(r3, inv, p, p_dash);
}

__device__ inline uint64_t d_mont_mul_sub(uint64_t a1, uint64_t b1,
                                          uint64_t a2, uint64_t b2, uint64_t p,
                                          uint64_t p_dash) {
  uint128_t t1 = (uint128_t)a1 * b1;
  uint128_t t2 = (uint128_t)a2 * b2;
  uint128_t t = t1 + ((uint128_t)p << 64) - t2;
  uint64_t m = (uint64_t)t * p_dash;
  uint64_t u = (t + (uint128_t)m * p) >> 64;
  int64_t maybe = u - p;
  return maybe < 0 ? u : (uint64_t)maybe;
}

// Build matrix for f_snd_trm on GPU, return dimension and prod_M
__device__ size_t d_f_snd_trm_build_matrix(const uint64_t *c, uint64_t m,
                                            const uint64_t *jk_prod_M,
                                            const uint64_t *nat_M,
                                            const uint64_t *nat_inv_M,
                                            uint64_t *A, uint64_t *prod_M_out,
                                            uint64_t p, uint64_t p_dash,
                                            uint64_t r) {
  uint64_t typ[32]; // Max m we support
  size_t r_cnt = 0;
  for (size_t i = 0; i < m; ++i) {
    if (c[i]) {
      typ[r_cnt] = i;
      ++r_cnt;
    }
  }

  uint64_t prod_M = r;

  for (size_t a = 0; a < r_cnt; ++a) {
    size_t i = typ[a];
    if (c[i] == 1)
      continue;

    uint64_t sum = 0;
    for (size_t b = 0; b < r_cnt; ++b) {
      size_t j = typ[b];
      uint64_t W = jk_prod_M[d_jk_pos(i, j, m)];
      sum = d_add_mod(sum, d_mont_mul(nat_M[c[j]], W, p, p_dash), p);
    }

    prod_M = d_mont_pow(sum, c[i] - 1, prod_M, p, p_dash);
  }

  prod_M = d_mont_mul(prod_M, nat_inv_M[c[0]], p, p_dash);
  size_t dim = r_cnt - 1;

  if (dim == 0) {
    *prod_M_out = prod_M;
    return 0;
  }

  for (size_t a = 1; a < r_cnt; ++a) {
    size_t i = typ[a];
    uint64_t W_del = jk_prod_M[m - i];
    uint64_t diag = d_mont_mul(nat_M[c[0]], W_del, p, p_dash);

    for (size_t b = 1; b < r_cnt; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      uint64_t W = jk_prod_M[d_jk_pos(i, j, m)];
      uint64_t v = d_mont_mul(nat_M[c[j]], W, p, p_dash);
      A[(a - 1) * dim + (b - 1)] = p - v;
      diag = d_add_mod(diag, v, p);
    }

    A[(a - 1) * dim + (a - 1)] = diag;
  }

  *prod_M_out = prod_M;
  return dim;
}

// Build matrix for jack_snd_trm on GPU
__device__ size_t d_jack_snd_trm_build_matrix(const uint64_t *c, uint64_t m,
                                               const uint64_t *jk_prod_M,
                                               const uint64_t *nat_M, uint64_t *A,
                                               uint64_t *prod_M_out, uint64_t p,
                                               uint64_t p_dash, uint64_t r) {
  uint64_t typ[32];
  size_t r_cnt = 0;
  for (size_t i = 0; i < m; ++i) {
    if (c[i]) {
      typ[r_cnt] = i;
      ++r_cnt;
    }
  }

  uint64_t prod_M = r;

  for (size_t a = 0; a < r_cnt; ++a) {
    size_t i = typ[a];
    uint64_t sum = r;
    for (size_t b = 0; b < r_cnt; ++b) {
      size_t j = typ[b];
      uint64_t w = jk_prod_M[d_jk_pos(i, j, m)];
      sum = d_add_mod(sum, d_mont_mul(nat_M[c[j]], w, p, p_dash), p);
    }
    prod_M = d_mont_pow(sum, c[i] - 1, prod_M, p, p_dash);
  }

  if (r_cnt <= 1) {
    *prod_M_out = prod_M;
    return 0;
  }

  size_t dim = r_cnt;
  for (size_t a = 0; a < r_cnt; ++a) {
    size_t i = typ[a];
    uint64_t diag = r;

    for (size_t b = 0; b < r_cnt; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      uint64_t w = jk_prod_M[d_jk_pos(i, j, m)];
      uint64_t v = d_mont_mul(nat_M[c[j]], w, p, p_dash);
      A[(a)*dim + (b)] = p - v;
      diag = d_add_mod(diag, v, p);
    }

    A[(a)*dim + (a)] = diag;
  }

  *prod_M_out = prod_M;
  return dim;
}

// Combined kernel: builds matrix from coefficient vector AND computes determinant
// Each thread processes one coefficient vector
// Template parameter MAX_DIM allows compile-time sizing of local array
template<int MAX_DIM>
__global__ void vec_det_kernel(const uint64_t *vecs, size_t n_vecs, uint64_t m,
                                bool is_jack_mode,
                                const uint64_t *d_jk_prod_M,
                                const uint64_t *d_nat_M,
                                const uint64_t *d_nat_inv_M, uint64_t *results,
                                uint64_t p, uint64_t p_dash, uint64_t r,
                                uint64_t r3) {
  int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_idx >= n_vecs)
    return;

  const uint64_t *c = &vecs[vec_idx * m];

  // Build matrix in fast local memory, sized exactly for this m
  uint64_t A[MAX_DIM * MAX_DIM];
  uint64_t prod_M;
  size_t dim;

  if (is_jack_mode) {
    dim = d_jack_snd_trm_build_matrix(c, m, d_jk_prod_M, d_nat_M, A, &prod_M,
                                       p, p_dash, r);
  } else {
    dim = d_f_snd_trm_build_matrix(c, m, d_jk_prod_M, d_nat_M, d_nat_inv_M, A,
                                    &prod_M, p, p_dash, r);
  }

  // If no matrix needed (dim == 0), just return prod_M
  if (dim == 0) {
    results[vec_idx] = prod_M;
    return;
  }

  // Compute determinant via Gaussian elimination
  uint64_t det = r, scaling_factor = r;

  for (size_t k = 0; k < dim; ++k) {
    // Find pivot
    size_t pivot_i = k;
    while (pivot_i < dim && A[pivot_i * dim + k] == 0)
      ++pivot_i;

    if (pivot_i == dim) {
      det = 0;
      break;
    }

    // Swap rows if needed
    if (pivot_i != k) {
      for (size_t j = 0; j < dim; ++j) {
        uint64_t tmp = A[k * dim + j];
        A[k * dim + j] = A[pivot_i * dim + j];
        A[pivot_i * dim + j] = tmp;
      }
      det = p - det;
    }

    uint64_t pivot = A[k * dim + k];
    det = d_mont_mul(det, pivot, p, p_dash);

    // Elimination
    for (size_t i = k + 1; i < dim; ++i) {
      scaling_factor = d_mont_mul(scaling_factor, pivot, p, p_dash);
      uint64_t multiplier = A[i * dim + k];
      for (size_t j = k; j < dim; ++j) {
        A[i * dim + j] = d_mont_mul_sub(A[i * dim + j], pivot, A[k * dim + j],
                                        multiplier, p, p_dash);
      }
    }
  }

  // Compute final result
  if (det != 0) {
    det = d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p,
                     p_dash);
  }

  // Multiply by prod_M
  uint64_t result = d_mont_mul(prod_M, det, p, p_dash);
  results[vec_idx] = result;
}

// OLD determinant computation kernel - one thread per matrix
// Each thread independently computes the determinant of one matrix
// Works directly in global memory - modifies the input!
__global__ void det_kernel(uint64_t *matrices, const size_t *dims,
                           const uint64_t *prod_Ms, uint64_t *results,
                           size_t n_matrices, size_t max_dim, uint64_t p,
                           uint64_t p_dash, uint64_t r, uint64_t r3) {
  int matrix_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (matrix_idx >= n_matrices)
    return;

  size_t dim = dims[matrix_idx];
  uint64_t prod_M = prod_Ms[matrix_idx];

  // Work directly in global memory to maximize occupancy
  size_t matrix_offset = matrix_idx * max_dim * max_dim;
  uint64_t *A = &matrices[matrix_offset];

  // Gaussian elimination
  uint64_t det = r, scaling_factor = r;

  for (size_t k = 0; k < dim; ++k) {
    // Find pivot
    size_t pivot_i = k;
    while (pivot_i < dim && A[pivot_i * max_dim + k] == 0)
      ++pivot_i;

    if (pivot_i == dim) {
      det = 0;
      break;
    }

    // Swap rows if needed
    if (pivot_i != k) {
      for (size_t j = 0; j < dim; ++j) {
        uint64_t tmp = A[k * max_dim + j];
        A[k * max_dim + j] = A[pivot_i * max_dim + j];
        A[pivot_i * max_dim + j] = tmp;
      }
      det = p - det; // Flip sign
    }

    uint64_t pivot = A[k * max_dim + k];
    det = d_mont_mul(det, pivot, p, p_dash);

    // Elimination
    for (size_t i = k + 1; i < dim; ++i) {
      scaling_factor = d_mont_mul(scaling_factor, pivot, p, p_dash);
      uint64_t multiplier = A[i * max_dim + k];
      for (size_t j = k; j < dim; ++j) {
        A[i * max_dim + j] = d_mont_mul_sub(A[i * max_dim + j], pivot,
                                            A[k * max_dim + j], multiplier, p, p_dash);
      }
    }
  }

  // Compute final result
  if (det != 0) {
    det = d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p,
                     p_dash);
  }

  // Multiply by prod_M as required by f_snd_trm pattern
  uint64_t result = d_mont_mul(prod_M, det, p, p_dash);
  results[matrix_idx] = result;
}

// Batch structure
struct det_batch_t {
  // Host data
  uint64_t *h_matrices;  // max_matrices * max_dim^2
  size_t *h_dims;        // max_matrices
  uint64_t *h_prod_Ms;   // max_matrices
  uint64_t *h_results;   // max_matrices

  // Device data
  uint64_t *d_matrices;
  size_t *d_dims;
  uint64_t *d_prod_Ms;
  uint64_t *d_results;

  // Batch parameters
  size_t max_matrices;
  size_t max_dim;
  size_t count; // current number of matrices

  // Montgomery parameters
  uint64_t p, p_dash, r, r3;
};

bool gpu_det_available(void) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

det_batch_t *det_batch_new(size_t max_matrices, size_t max_dim, uint64_t p,
                           uint64_t p_dash, uint64_t r, uint64_t r3) {
  det_batch_t *batch = (det_batch_t *)malloc(sizeof(det_batch_t));
  assert(batch);

  batch->max_matrices = max_matrices;
  batch->max_dim = max_dim;
  batch->count = 0;
  batch->p = p;
  batch->p_dash = p_dash;
  batch->r = r;
  batch->r3 = r3;

  // Allocate host memory
  size_t matrix_size = max_matrices * max_dim * max_dim * sizeof(uint64_t);
  batch->h_matrices = (uint64_t *)malloc(matrix_size);
  batch->h_dims = (size_t *)malloc(max_matrices * sizeof(size_t));
  batch->h_prod_Ms = (uint64_t *)malloc(max_matrices * sizeof(uint64_t));
  batch->h_results = (uint64_t *)malloc(max_matrices * sizeof(uint64_t));
  assert(batch->h_matrices && batch->h_dims && batch->h_prod_Ms &&
         batch->h_results);

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&batch->d_matrices, matrix_size));
  CUDA_CHECK(cudaMalloc(&batch->d_dims, max_matrices * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_prod_Ms, max_matrices * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_results, max_matrices * sizeof(uint64_t)));

  return batch;
}

size_t det_batch_add(det_batch_t *batch, const uint64_t *matrix, size_t dim,
                     uint64_t prod_M) {
  assert(batch->count < batch->max_matrices);
  assert(dim <= batch->max_dim);

  size_t idx = batch->count++;

  // Copy matrix (padded to max_dim x max_dim)
  size_t offset = idx * batch->max_dim * batch->max_dim;
  memset(&batch->h_matrices[offset], 0,
         batch->max_dim * batch->max_dim * sizeof(uint64_t));
  for (size_t i = 0; i < dim; ++i) {
    for (size_t j = 0; j < dim; ++j) {
      batch->h_matrices[offset + i * batch->max_dim + j] =
          matrix[i * dim + j];
    }
  }

  batch->h_dims[idx] = dim;
  batch->h_prod_Ms[idx] = prod_M;

  return idx;
}

void det_batch_compute(det_batch_t *batch) {
  if (batch->count == 0)
    return;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Copy data to device
  cudaEventRecord(start);
  size_t matrix_size =
      batch->count * batch->max_dim * batch->max_dim * sizeof(uint64_t);
  CUDA_CHECK(cudaMemcpy(batch->d_matrices, batch->h_matrices, matrix_size,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_dims, batch->h_dims,
                        batch->count * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_prod_Ms, batch->h_prod_Ms,
                        batch->count * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float h2d_ms;
  cudaEventElapsedTime(&h2d_ms, start, stop);

  // Launch kernel - one thread per matrix
  // Higher block size for better occupancy (no local memory pressure now)
  cudaEventRecord(start);
  int block_size = 256; // Threads per block
  int num_blocks = (batch->count + block_size - 1) / block_size;

  det_kernel<<<num_blocks, block_size>>>(
      batch->d_matrices, batch->d_dims, batch->d_prod_Ms, batch->d_results,
      batch->count, batch->max_dim, batch->p, batch->p_dash, batch->r,
      batch->r3);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float kernel_ms;
  cudaEventElapsedTime(&kernel_ms, start, stop);

  // Copy results back
  cudaEventRecord(start);
  CUDA_CHECK(cudaMemcpy(batch->h_results, batch->d_results,
                        batch->count * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float d2h_ms;
  cudaEventElapsedTime(&d2h_ms, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

uint64_t det_batch_get(const det_batch_t *batch, size_t idx) {
  assert(idx < batch->count);
  return batch->h_results[idx];
}

void det_batch_clear(det_batch_t *batch) { batch->count = 0; }

void det_batch_free(det_batch_t *batch) {
  if (!batch)
    return;

  free(batch->h_matrices);
  free(batch->h_dims);
  free(batch->h_prod_Ms);
  free(batch->h_results);

  CUDA_CHECK(cudaFree(batch->d_matrices));
  CUDA_CHECK(cudaFree(batch->d_dims));
  CUDA_CHECK(cudaFree(batch->d_prod_Ms));
  CUDA_CHECK(cudaFree(batch->d_results));

  free(batch);
}

// ============================================================================
// Vector-based batch API (builds matrices on GPU)
// ============================================================================

// Number of CUDA streams for pipelining
#define NUM_STREAMS 4

struct vec_batch_t {
  // Host data (pinned for async transfers)
  uint64_t *h_vecs;    // max_vecs * m
  uint64_t *h_results; // max_vecs

  // Device data
  uint64_t *d_vecs;
  uint64_t *d_jk_prod_M;
  uint64_t *d_nat_M;
  uint64_t *d_nat_inv_M;
  uint64_t *d_results;

  // CUDA streams for pipelining
  cudaStream_t streams[NUM_STREAMS];

  // Batch parameters
  size_t max_vecs;
  size_t count;
  uint64_t n;  // For nat_M sizing
  uint64_t m;
  bool is_jack_mode;

  // Montgomery parameters
  uint64_t p, p_dash, r, r3;
};

vec_batch_t *vec_batch_new(size_t max_vecs, uint64_t n, uint64_t m, uint64_t p,
                            uint64_t p_dash, uint64_t r, uint64_t r3,
                            const uint64_t *jk_prod_M, const uint64_t *nat_M,
                            const uint64_t *nat_inv_M, bool is_jack_mode) {
  vec_batch_t *batch = (vec_batch_t *)malloc(sizeof(vec_batch_t));
  assert(batch);

  batch->max_vecs = max_vecs;
  batch->count = 0;
  batch->n = n;
  batch->m = m;
  batch->p = p;
  batch->p_dash = p_dash;
  batch->r = r;
  batch->r3 = r3;
  batch->is_jack_mode = is_jack_mode;

  // Allocate pinned host memory for faster async transfers
  CUDA_CHECK(cudaMallocHost(&batch->h_vecs, max_vecs * m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMallocHost(&batch->h_results, max_vecs * sizeof(uint64_t)));

  // Create CUDA streams for pipelining
  for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamCreate(&batch->streams[i]));
  }

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&batch->d_vecs, max_vecs * m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_jk_prod_M, m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_nat_M, (n + 1) * sizeof(uint64_t)));  // FIX: n not m!
  if (!is_jack_mode) {
    CUDA_CHECK(cudaMalloc(&batch->d_nat_inv_M, (n + 1) * sizeof(uint64_t)));  // FIX: n not m!
  } else {
    batch->d_nat_inv_M = NULL;
  }
  CUDA_CHECK(cudaMalloc(&batch->d_results, max_vecs * sizeof(uint64_t)));

  // Copy constant data to device
  CUDA_CHECK(cudaMemcpy(batch->d_jk_prod_M, jk_prod_M, m * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_nat_M, nat_M, (n + 1) * sizeof(uint64_t),  // FIX: n not m!
                        cudaMemcpyHostToDevice));
  if (!is_jack_mode) {
    CUDA_CHECK(cudaMemcpy(batch->d_nat_inv_M, nat_inv_M,
                          (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));  // FIX: n not m!
  }

  return batch;
}

size_t vec_batch_add(vec_batch_t *batch, const uint64_t *vec) {
  assert(batch->count < batch->max_vecs);
  size_t idx = batch->count++;
  memcpy(&batch->h_vecs[idx * batch->m], vec, batch->m * sizeof(uint64_t));
  return idx;
}

// Helper macro to launch appropriate kernel based on m
#define LAUNCH_KERNEL_ON_STREAM(MAX_DIM, stream, offset, count) \
  vec_det_kernel<MAX_DIM><<<num_blocks, block_size, 0, stream>>>( \
      batch->d_vecs + (offset) * batch->m, (count), batch->m, batch->is_jack_mode, \
      batch->d_jk_prod_M, batch->d_nat_M, batch->d_nat_inv_M, \
      batch->d_results + (offset), batch->p, batch->p_dash, batch->r, batch->r3)

void vec_batch_compute(vec_batch_t *batch) {
  if (batch->count == 0)
    return;

  // Split batch into sub-batches for streaming
  // Each stream handles one sub-batch, allowing H2D, kernel, and D2H to overlap
  size_t vecs_per_stream = (batch->count + NUM_STREAMS - 1) / NUM_STREAMS;
  int block_size = 256;

  for (int i = 0; i < NUM_STREAMS; i++) {
    size_t offset = i * vecs_per_stream;
    if (offset >= batch->count)
      break;

    size_t count = (offset + vecs_per_stream > batch->count)
                   ? (batch->count - offset)
                   : vecs_per_stream;

    cudaStream_t stream = batch->streams[i];

    // Async copy H2D for this sub-batch
    size_t vecs_size = count * batch->m * sizeof(uint64_t);
    CUDA_CHECK(cudaMemcpyAsync(batch->d_vecs + offset * batch->m,
                               batch->h_vecs + offset * batch->m,
                               vecs_size,
                               cudaMemcpyHostToDevice,
                               stream));

    // Launch kernel on this stream for this sub-batch
    int num_blocks = (count + block_size - 1) / block_size;

    // Dispatch to kernel with compile-time sized array based on m
    if (batch->m <= 13) {
      LAUNCH_KERNEL_ON_STREAM(13, stream, offset, count);
    } else if (batch->m <= 17) {
      LAUNCH_KERNEL_ON_STREAM(17, stream, offset, count);
    } else if (batch->m <= 21) {
      LAUNCH_KERNEL_ON_STREAM(21, stream, offset, count);
    } else if (batch->m <= 25) {
      LAUNCH_KERNEL_ON_STREAM(25, stream, offset, count);
    } else {
      LAUNCH_KERNEL_ON_STREAM(32, stream, offset, count);
    }

    CUDA_CHECK(cudaGetLastError());

    // Async copy D2H for results of this sub-batch
    CUDA_CHECK(cudaMemcpyAsync(batch->h_results + offset,
                               batch->d_results + offset,
                               count * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost,
                               stream));
  }

  // Wait for all streams to complete
  for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamSynchronize(batch->streams[i]));
  }
}

uint64_t vec_batch_get(const vec_batch_t *batch, size_t idx) {
  assert(idx < batch->count);
  return batch->h_results[idx];
}

void vec_batch_clear(vec_batch_t *batch) { batch->count = 0; }

void vec_batch_free(vec_batch_t *batch) {
  if (!batch)
    return;

  // Destroy CUDA streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamDestroy(batch->streams[i]));
  }

  // Free pinned host memory
  CUDA_CHECK(cudaFreeHost(batch->h_vecs));
  CUDA_CHECK(cudaFreeHost(batch->h_results));

  // Free device memory
  CUDA_CHECK(cudaFree(batch->d_vecs));
  CUDA_CHECK(cudaFree(batch->d_jk_prod_M));
  CUDA_CHECK(cudaFree(batch->d_nat_M));
  if (batch->d_nat_inv_M)
    CUDA_CHECK(cudaFree(batch->d_nat_inv_M));
  CUDA_CHECK(cudaFree(batch->d_results));

  free(batch);
}
