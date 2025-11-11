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

// Power cache parameters (must match source_process.c)
#define POW_CACHE_SPLIT 6
#define POW_CACHE_DIVISOR (1 << POW_CACHE_SPLIT)

// Number of CUDA streams for pipelining
#define NUM_STREAMS 4

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

// Device-side fast_pow_2 using rs cache
__device__ inline uint64_t d_fast_pow_2(const uint64_t *d_rs, uint64_t pow,
                                        uint64_t p, uint64_t p_dash) {
  uint64_t r_pow = pow / 64;
  uint64_t remain = pow % 64;
  uint64_t pow2 = 1UL << remain;
  return d_mont_mul(pow2, d_rs[r_pow + 2], p, p_dash);
}

// Device-side jk_sums_pow using split power cache
__device__ inline uint64_t d_jk_sums_pow(const uint64_t *d_jk_sums_pow_upper_M,
                                          const uint64_t *d_jk_sums_pow_lower_M,
                                          uint64_t diff, uint64_t pow,
                                          size_t m_half, uint64_t p, uint64_t p_dash) {
  uint64_t upper_index = pow >> POW_CACHE_SPLIT;
  uint64_t lower_index = pow & (POW_CACHE_DIVISOR - 1);

  uint64_t upper_index_full = upper_index * m_half + diff;
  uint64_t lower_index_full = lower_index * m_half + diff;

  return d_mont_mul(d_jk_sums_pow_upper_M[upper_index_full],
                    d_jk_sums_pow_lower_M[lower_index_full], p, p_dash);
}

// Device-side multinomial coefficient computation
__device__ inline uint64_t d_multinomial_mod_p(const uint64_t *d_fact_M,
                                                const uint64_t *d_fact_inv_M,
                                                const uint64_t *vec, uint64_t m,
                                                uint64_t n_args, uint64_t p,
                                                uint64_t p_dash) {
  uint64_t coeff = d_fact_M[n_args - 1];
  for (size_t i = 0; i < m; ++i) {
    coeff = d_mont_mul(coeff, d_fact_inv_M[vec[i]], p, p_dash);
  }
  return coeff;
}

// Device-side f_fst_trm computation
__device__ inline uint64_t d_f_fst_trm(const uint64_t *vec, uint64_t m,
                                        size_t m_half, const uint64_t *d_rs,
                                        const uint64_t *d_jk_sums_pow_upper_M,
                                        const uint64_t *d_jk_sums_pow_lower_M,
                                        uint64_t p, uint64_t p_dash) {
  uint64_t e = 0;
  uint64_t pows[32];  // Assume m <= 32
  for (size_t i = 0; i < m; ++i) {
    pows[i] = 0;
  }

  for (size_t a = 0; a < m; ++a) {
    uint64_t ca = vec[a];
    if (!ca)
      continue;

    e += (ca * (ca - 1));

    for (size_t b = a + 1; b < m; ++b) {
      uint64_t cb = vec[b];
      uint64_t diff = b - a;
      pows[diff] += ca * cb;
    }
  }

  uint64_t acc = d_fast_pow_2(d_rs, e / 2, p, p_dash);

  for (size_t i = 1; i < m_half; i++) {
    uint64_t pow_val = d_jk_sums_pow(d_jk_sums_pow_upper_M, d_jk_sums_pow_lower_M,
                                      i, pows[i] + pows[m - i], m_half, p, p_dash);
    acc = d_mont_mul(acc, pow_val, p, p_dash);
  }

  return acc;
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

// Comprehensive kernel: computes full david() or jack() result on GPU
// Each thread processes one coefficient vector and produces final result
template<int MAX_DIM>
__global__ void vec_full_kernel(const uint64_t *vecs, size_t n_vecs, uint64_t m,
                                 uint64_t n, uint64_t n_args, size_t m_half,
                                 bool is_jack_mode,
                                 const uint64_t *d_jk_prod_M,
                                 const uint64_t *d_nat_M,
                                 const uint64_t *d_nat_inv_M,
                                 const uint64_t *d_ws_M,
                                 const uint64_t *d_jk_sums_M,
                                 const uint64_t *d_jk_sums_pow_lower_M,
                                 const uint64_t *d_jk_sums_pow_upper_M,
                                 const uint64_t *d_rs,
                                 const uint64_t *d_fact_M,
                                 const uint64_t *d_fact_inv_M,
                                 uint64_t *results,
                                 uint64_t p, uint64_t p_dash, uint64_t r,
                                 uint64_t r3) {
  int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_idx >= n_vecs)
    return;

  const uint64_t *vec = &vecs[vec_idx * m];

  // Step 1: Compute f_fst_trm
  uint64_t f_fst_result = d_f_fst_trm(vec, m, m_half, d_rs,
                                       d_jk_sums_pow_upper_M,
                                       d_jk_sums_pow_lower_M, p, p_dash);

  // Step 2: Compute f_snd_trm (build matrix + compute determinant)
  uint64_t A[MAX_DIM * MAX_DIM];
  uint64_t prod_M;
  size_t dim;

  if (is_jack_mode) {
    dim = d_jack_snd_trm_build_matrix(vec, m, d_jk_prod_M, d_nat_M, A, &prod_M,
                                       p, p_dash, r);
  } else {
    dim = d_f_snd_trm_build_matrix(vec, m, d_jk_prod_M, d_nat_M, d_nat_inv_M, A,
                                    &prod_M, p, p_dash, r);
  }

  uint64_t f_snd_result;
  if (dim == 0) {
    f_snd_result = prod_M;
  } else {
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

    // Compute final determinant
    if (det != 0) {
      det = d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p, p_dash);
    }

    f_snd_result = d_mont_mul(prod_M, det, p, p_dash);
  }

  // Step 3: Multiply f_fst_result and f_snd_result to get f_0
  uint64_t f_0 = d_mont_mul(f_fst_result, f_snd_result, p, p_dash);

  // Step 4: Compute multinomial coefficient
  uint64_t coeff_baseline = d_multinomial_mod_p(d_fact_M, d_fact_inv_M, vec, m,
                                                 n_args, p, p_dash);

  // Step 5: Loop over rotations (david or jack)
  uint64_t ret = 0;
  for (size_t r_idx = 0; r_idx < m; ++r_idx) {
    if (vec[r_idx] == 0)
      continue;

    uint64_t coeff = d_mont_mul(coeff_baseline, d_nat_M[vec[r_idx]], p, p_dash);

    uint64_t f_n;
    if (is_jack_mode) {
      // Jack mode
      f_n = d_mont_mul(coeff, f_0, p, p_dash);
    } else {
      // David mode
      size_t idx = (2 * r_idx) % m;
      f_n = d_mont_mul(coeff,
                       d_mont_mul(f_0, d_ws_M[idx ? m - idx : 0], p, p_dash),
                       p, p_dash);
    }

    ret = d_add_mod(ret, f_n, p);
  }

  // Step 6: Apply final scaling for jack mode
  if (is_jack_mode) {
    ret = d_mont_mul(ret, d_nat_M[n - 1], p, p_dash);
    ret = d_mont_mul(ret, d_nat_M[n - 1], p, p_dash);
  }

  results[vec_idx] = ret;
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
  if (err != cudaSuccess || device_count == 0)
    return false;

  // Print GPU info on first call
  static bool printed = false;
  if (!printed) {
    fprintf(stderr, "Found %d GPU%s:\n", device_count, device_count > 1 ? "s" : "");
    for (int i = 0; i < device_count; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      fprintf(stderr, "  GPU %d: %s (SM count: %d, Max threads per block: %d)\n",
             i, prop.name, prop.multiProcessorCount, prop.maxThreadsPerBlock);
    }
    printed = true;
  }

  return true;
}

int gpu_det_device_count(void) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    return 0;
  return device_count;
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

// Shared GPU context - constant lookup tables shared across all batches on a GPU
struct gpu_shared_ctx_t {
  int device_id;

  // Device data - matrix computation lookups (constant per prime)
  uint64_t *d_jk_prod_M;
  uint64_t *d_nat_M;
  uint64_t *d_nat_inv_M;  // NULL in jack mode

  // Device data - full computation lookup tables (constant per prime)
  uint64_t *d_ws_M;           // m
  uint64_t *d_jk_sums_M;      // m
  uint64_t *d_jk_sums_pow_lower_M;  // POW_CACHE_DIVISOR * m_half
  uint64_t *d_jk_sums_pow_upper_M;  // POW_CACHE_DIVISOR * m_half
  uint64_t *d_rs;             // n_rs
  uint64_t *d_fact_M;         // n+1
  uint64_t *d_fact_inv_M;     // n+1

  // Parameters (constant per prime)
  uint64_t n;
  uint64_t n_args;
  uint64_t m;
  size_t m_half;
  size_t n_rs;
  bool is_jack_mode;

  // Montgomery parameters (constant per prime)
  uint64_t p, p_dash, r, r3;
};

struct vec_batch_t {
  // Reference to shared context (constant lookup tables)
  gpu_shared_ctx_t *shared;

  // Host data (pinned for async transfers)
  uint64_t *h_vecs;    // max_vecs * m
  uint64_t *h_results; // max_vecs

  // Device data - per-batch varying data only
  uint64_t *d_vecs;
  uint64_t *d_results;

  // CUDA streams for pipelining
  cudaStream_t streams[NUM_STREAMS];

  // Batch state
  size_t max_vecs;
  size_t count;
};

gpu_shared_ctx_t *gpu_shared_ctx_new(int device_id, uint64_t n, uint64_t n_args, uint64_t m,
                                      uint64_t p, uint64_t p_dash, uint64_t r, uint64_t r3,
                                      const uint64_t *jk_prod_M, const uint64_t *nat_M,
                                      const uint64_t *nat_inv_M, const uint64_t *ws_M,
                                      const uint64_t *jk_sums_M, const uint64_t *jk_sums_pow_lower_M,
                                      const uint64_t *jk_sums_pow_upper_M, const uint64_t *rs,
                                      const uint64_t *fact_M, const uint64_t *fact_inv_M,
                                      size_t m_half, size_t n_rs, bool is_jack_mode) {
  gpu_shared_ctx_t *ctx = (gpu_shared_ctx_t *)malloc(sizeof(gpu_shared_ctx_t));
  assert(ctx);

  // Set device
  cudaSetDevice(device_id);
  ctx->device_id = device_id;

  // Store parameters
  ctx->n = n;
  ctx->n_args = n_args;
  ctx->m = m;
  ctx->m_half = m_half;
  ctx->n_rs = n_rs;
  ctx->p = p;
  ctx->p_dash = p_dash;
  ctx->r = r;
  ctx->r3 = r3;
  ctx->is_jack_mode = is_jack_mode;

  // Allocate device memory for matrix computation lookups
  CUDA_CHECK(cudaMalloc(&ctx->d_jk_prod_M, m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_nat_M, (n + 1) * sizeof(uint64_t)));
  if (!is_jack_mode) {
    CUDA_CHECK(cudaMalloc(&ctx->d_nat_inv_M, (n + 1) * sizeof(uint64_t)));
  } else {
    ctx->d_nat_inv_M = NULL;
  }

  // Allocate device memory for full computation lookup tables
  CUDA_CHECK(cudaMalloc(&ctx->d_ws_M, m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_jk_sums_M, m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_jk_sums_pow_lower_M, POW_CACHE_DIVISOR * m_half * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_jk_sums_pow_upper_M, POW_CACHE_DIVISOR * m_half * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_rs, n_rs * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_fact_M, (n + 1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_fact_inv_M, (n + 1) * sizeof(uint64_t)));

  // Copy constant data to device - matrix computation
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_prod_M, jk_prod_M, m * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_nat_M, nat_M, (n + 1) * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  if (!is_jack_mode) {
    CUDA_CHECK(cudaMemcpy(ctx->d_nat_inv_M, nat_inv_M,
                          (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  }

  // Copy constant data to device - full computation lookup tables
  CUDA_CHECK(cudaMemcpy(ctx->d_ws_M, ws_M, m * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_sums_M, jk_sums_M, m * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_sums_pow_lower_M, jk_sums_pow_lower_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_sums_pow_upper_M, jk_sums_pow_upper_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_rs, rs, n_rs * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_fact_M, fact_M, (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_fact_inv_M, fact_inv_M, (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

  return ctx;
}

void gpu_shared_ctx_free(gpu_shared_ctx_t *ctx) {
  if (!ctx) return;

  cudaSetDevice(ctx->device_id);

  cudaFree(ctx->d_jk_prod_M);
  cudaFree(ctx->d_nat_M);
  if (ctx->d_nat_inv_M) {
    cudaFree(ctx->d_nat_inv_M);
  }
  cudaFree(ctx->d_ws_M);
  cudaFree(ctx->d_jk_sums_M);
  cudaFree(ctx->d_jk_sums_pow_lower_M);
  cudaFree(ctx->d_jk_sums_pow_upper_M);
  cudaFree(ctx->d_rs);
  cudaFree(ctx->d_fact_M);
  cudaFree(ctx->d_fact_inv_M);

  free(ctx);
}

vec_batch_t *vec_batch_new(size_t max_vecs, gpu_shared_ctx_t *shared_ctx) {
  vec_batch_t *batch = (vec_batch_t *)malloc(sizeof(vec_batch_t));
  assert(batch);

  // Set device
  cudaSetDevice(shared_ctx->device_id);

  // Reference shared context
  batch->shared = shared_ctx;

  batch->max_vecs = max_vecs;
  batch->count = 0;

  // Allocate pinned host memory for faster async transfers
  CUDA_CHECK(cudaMallocHost(&batch->h_vecs, max_vecs * shared_ctx->m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMallocHost(&batch->h_results, max_vecs * sizeof(uint64_t)));

  // Create CUDA streams for pipelining
  for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamCreate(&batch->streams[i]));
  }

  // Allocate device memory - only per-batch varying data
  CUDA_CHECK(cudaMalloc(&batch->d_vecs, max_vecs * shared_ctx->m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_results, max_vecs * sizeof(uint64_t)));

  return batch;
}

size_t vec_batch_add(vec_batch_t *batch, const uint64_t *vec) {
  assert(batch->count < batch->max_vecs);
  size_t idx = batch->count++;
  memcpy(&batch->h_vecs[idx * batch->shared->m], vec, batch->shared->m * sizeof(uint64_t));
  return idx;
}

// Helper macro to launch appropriate kernel based on m
#define LAUNCH_KERNEL_ON_STREAM(MAX_DIM, stream, offset, count) \
  vec_det_kernel<MAX_DIM><<<num_blocks, block_size, 0, stream>>>( \
      batch->d_vecs + (offset) * batch->shared->m, (count), batch->shared->m, batch->shared->is_jack_mode, \
      batch->shared->d_jk_prod_M, batch->shared->d_nat_M, batch->shared->d_nat_inv_M, \
      batch->d_results + (offset), batch->shared->p, batch->shared->p_dash, batch->shared->r, batch->shared->r3)

// Helper macro to launch comprehensive kernel (full david/jack computation)
#define LAUNCH_FULL_KERNEL_ON_STREAM(MAX_DIM, stream, offset, count) \
  vec_full_kernel<MAX_DIM><<<num_blocks, block_size, 0, stream>>>( \
      batch->d_vecs + (offset) * batch->shared->m, (count), batch->shared->m, \
      batch->shared->n, batch->shared->n_args, batch->shared->m_half, batch->shared->is_jack_mode, \
      batch->shared->d_jk_prod_M, batch->shared->d_nat_M, batch->shared->d_nat_inv_M, \
      batch->shared->d_ws_M, batch->shared->d_jk_sums_M, \
      batch->shared->d_jk_sums_pow_lower_M, batch->shared->d_jk_sums_pow_upper_M, \
      batch->shared->d_rs, batch->shared->d_fact_M, batch->shared->d_fact_inv_M, \
      batch->d_results + (offset), batch->shared->p, batch->shared->p_dash, batch->shared->r, batch->shared->r3)

// Launch async GPU compute (non-blocking)
void vec_batch_compute_async(vec_batch_t *batch) {
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
    size_t vecs_size = count * batch->shared->m * sizeof(uint64_t);
    CUDA_CHECK(cudaMemcpyAsync(batch->d_vecs + offset * batch->shared->m,
                               batch->h_vecs + offset * batch->shared->m,
                               vecs_size,
                               cudaMemcpyHostToDevice,
                               stream));

    // Launch kernel on this stream for this sub-batch
    int num_blocks = (count + block_size - 1) / block_size;

    // Dispatch to comprehensive kernel with compile-time sized array based on m
    if (batch->shared->m <= 13) {
      LAUNCH_FULL_KERNEL_ON_STREAM(13, stream, offset, count);
    } else if (batch->shared->m <= 17) {
      LAUNCH_FULL_KERNEL_ON_STREAM(17, stream, offset, count);
    } else if (batch->shared->m <= 21) {
      LAUNCH_FULL_KERNEL_ON_STREAM(21, stream, offset, count);
    } else if (batch->shared->m <= 25) {
      LAUNCH_FULL_KERNEL_ON_STREAM(25, stream, offset, count);
    } else {
      LAUNCH_FULL_KERNEL_ON_STREAM(32, stream, offset, count);
    }

    CUDA_CHECK(cudaGetLastError());

    // Async copy D2H for results of this sub-batch
    CUDA_CHECK(cudaMemcpyAsync(batch->h_results + offset,
                               batch->d_results + offset,
                               count * sizeof(uint64_t),
                               cudaMemcpyDeviceToHost,
                               stream));
  }
  // Returns immediately - GPU work continues asynchronously
}

// Wait for async compute to complete
void vec_batch_wait(vec_batch_t *batch) {
  for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamSynchronize(batch->streams[i]));
  }
}

// Synchronous compute (launch + wait)
void vec_batch_compute(vec_batch_t *batch) {
  vec_batch_compute_async(batch);
  vec_batch_wait(batch);
}

uint64_t vec_batch_get(const vec_batch_t *batch, size_t idx) {
  assert(idx < batch->count);
  return batch->h_results[idx];
}

void vec_batch_clear(vec_batch_t *batch) { batch->count = 0; }

void vec_batch_free(vec_batch_t *batch) {
  if (!batch)
    return;

  // Set device
  cudaSetDevice(batch->shared->device_id);

  // Destroy CUDA streams
  for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamDestroy(batch->streams[i]));
  }

  // Free pinned host memory
  CUDA_CHECK(cudaFreeHost(batch->h_vecs));
  CUDA_CHECK(cudaFreeHost(batch->h_results));

  // Free device memory - only per-batch varying data
  // Lookup tables are owned by shared context
  CUDA_CHECK(cudaFree(batch->d_vecs));
  CUDA_CHECK(cudaFree(batch->d_results));

  free(batch);
}
