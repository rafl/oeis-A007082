#include "debug.h"
#include "gpu_det.h"

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

typedef unsigned __int128 uint128_t;

// Helper function for jk_pos on device
template <size_t M> __device__ inline size_t d_jk_pos(size_t j, size_t k) {
  int64_t result = k - j;
  return result >= 0 ? (uint64_t)result : result + M;
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

__device__ inline uint64_t d_mont_mul_sub(uint64_t a1, uint64_t b1, uint64_t a2,
                                          uint64_t b2, uint64_t p,
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
template <size_t M_HALF>
__device__ inline uint64_t d_jk_sums_pow(const uint64_t *d_jk_sums_pow_upper_M,
                                         const uint64_t *d_jk_sums_pow_lower_M,
                                         uint64_t diff, uint64_t pow,
                                         uint64_t p, uint64_t p_dash) {
  uint64_t upper_index = pow >> POW_CACHE_SPLIT;
  uint64_t lower_index = pow & (POW_CACHE_DIVISOR - 1);

  uint64_t upper_index_full = upper_index * M_HALF + diff;
  uint64_t lower_index_full = lower_index * M_HALF + diff;

  return d_mont_mul(d_jk_sums_pow_upper_M[upper_index_full],
                    d_jk_sums_pow_lower_M[lower_index_full], p, p_dash);
}

// Device-side multinomial coefficient computation
template <size_t M>
__device__ inline uint64_t
d_multinomial_mod_p(const uint64_t *d_fact_M, const uint64_t *d_fact_inv_M,
                    const uint64_t *vec, uint64_t n_args, uint64_t p,
                    uint64_t p_dash) {
  uint64_t coeff = d_fact_M[n_args - 1];
  for (size_t i = 0; i < M; ++i) {
    coeff = d_mont_mul(coeff, d_fact_inv_M[vec[i]], p, p_dash);
  }
  return coeff;
}

// Device-side f_fst_trm computation
template <size_t M, size_t M_HALF>
__device__ inline uint64_t d_f_fst_trm(const uint64_t *vec,
                                       const uint64_t *d_rs,
                                       const uint64_t *d_jk_sums_pow_upper_M,
                                       const uint64_t *d_jk_sums_pow_lower_M,
                                       uint64_t p, uint64_t p_dash) {
  uint64_t e = 0;
  // assert(m <= M);
  uint64_t pows[M];
  for (size_t i = 0; i < M; ++i) {
    pows[i] = 0;
  }

  for (size_t a = 0; a < M; ++a) {
    uint64_t ca = vec[a];
    if (!ca)
      continue;

    e += (ca * (ca - 1));

    for (size_t b = a + 1; b < M; ++b) {
      uint64_t cb = vec[b];
      uint64_t diff = b - a;
      pows[diff] += ca * cb;
    }
  }

  uint64_t acc = d_fast_pow_2(d_rs, e / 2, p, p_dash);

  for (size_t i = 1; i < M_HALF; i++) {
    uint64_t pow_val =
        d_jk_sums_pow<M_HALF>(d_jk_sums_pow_upper_M, d_jk_sums_pow_lower_M, i,
                              pows[i] + pows[M - i], p, p_dash);
    acc = d_mont_mul(acc, pow_val, p, p_dash);
  }

  return acc;
}

// Build matrix for f_snd_trm on GPU, return dimension and prod_M
template <size_t M>
__device__ size_t d_f_snd_trm_build_matrix(
    const uint64_t *c, const uint64_t *jk_prod_M, const uint64_t *nat_M,
    const uint64_t *nat_inv_M, uint64_t *A, uint64_t *prod_M_out, uint64_t p,
    uint64_t p_dash, uint64_t r) {
  uint64_t typ[M];
  size_t r_cnt = 0;
  for (size_t i = 0; i < M; ++i) {
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
      uint64_t W = jk_prod_M[d_jk_pos<M>(i, j)];
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
    uint64_t W_del = jk_prod_M[M - i];
    uint64_t diag = d_mont_mul(nat_M[c[0]], W_del, p, p_dash);

    for (size_t b = 1; b < r_cnt; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      uint64_t W = jk_prod_M[d_jk_pos<M>(i, j)];
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
template <size_t M>
__device__ size_t d_jack_snd_trm_build_matrix(const uint64_t *c,
                                              const uint64_t *jk_prod_M,
                                              const uint64_t *nat_M,
                                              uint64_t *A, uint64_t *prod_M_out,
                                              uint64_t p, uint64_t p_dash,
                                              uint64_t r) {
  uint64_t typ[M];
  size_t dim = 0;
  for (size_t i = 0; i < M; ++i) {
    if (c[i]) {
      typ[dim] = i;
      ++dim;
    }
  }

  uint64_t prod_M = r;

  for (size_t a = 0; a < dim; ++a) {
    size_t i = typ[a];
    uint64_t sum = r;
    for (size_t b = 0; b < dim; ++b) {
      size_t j = typ[b];
      uint64_t w = jk_prod_M[d_jk_pos<M>(i, j)];
      sum = d_add_mod(sum, d_mont_mul(nat_M[c[j]], w, p, p_dash), p);
    }
    prod_M = d_mont_pow(sum, c[i] - 1, prod_M, p, p_dash);
  }

  if (dim <= 1) {
    *prod_M_out = prod_M;
    return 0;
  }

  for (size_t a = 0; a < dim; ++a) {
    size_t i = typ[a];
    uint64_t diag = r;

    for (size_t b = 0; b < dim; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      uint64_t w = jk_prod_M[d_jk_pos<M>(i, j)];
      uint64_t v = d_mont_mul(nat_M[c[j]], w, p, p_dash);
      A[(a)*dim + (b)] = p - v;
      diag = d_add_mod(diag, v, p);
    }

    A[(a)*dim + (a)] = diag;
  }

  *prod_M_out = prod_M;
  return dim;
}

// Comprehensive kernel: computes full david() or jack() result on GPU
// Each thread processes one coefficient vector and produces final result
template <size_t M, size_t M_HALF>
__global__ void
vec_full_kernel(const uint64_t *vecs, size_t n_vecs, uint64_t n,
                uint64_t n_args, bool is_jack_mode, const uint64_t *d_jk_prod_M,
                const uint64_t *d_nat_M, const uint64_t *d_nat_inv_M,
                const uint64_t *d_ws_M, const uint64_t *d_jk_sums_pow_lower_M,
                const uint64_t *d_jk_sums_pow_upper_M, const uint64_t *d_rs,
                const uint64_t *d_fact_M, const uint64_t *d_fact_inv_M,
                uint64_t *results, uint64_t p, uint64_t p_dash, uint64_t r,
                uint64_t r3) {
  size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_idx >= n_vecs)
    return;

  const uint64_t *vec = &vecs[vec_idx * M];

  // Step 1: Compute f_fst_trm
  uint64_t f_fst_result = d_f_fst_trm<M, M_HALF>(
      vec, d_rs, d_jk_sums_pow_upper_M, d_jk_sums_pow_lower_M, p, p_dash);

  // Step 2: Compute f_snd_trm (build matrix + compute determinant)
  uint64_t A[M * M];
  uint64_t prod_M;
  size_t dim;

  if (is_jack_mode) {
    dim = d_jack_snd_trm_build_matrix<M>(vec, d_jk_prod_M, d_nat_M, A, &prod_M,
                                         p, p_dash, r);
  } else {
    dim = d_f_snd_trm_build_matrix<M>(vec, d_jk_prod_M, d_nat_M, d_nat_inv_M, A,
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
      det =
          d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p, p_dash);
    }

    f_snd_result = d_mont_mul(prod_M, det, p, p_dash);
  }

  // Step 3: Multiply f_fst_result and f_snd_result to get f_0
  uint64_t f_0 = d_mont_mul(f_fst_result, f_snd_result, p, p_dash);

  // Step 4: Compute multinomial coefficient
  uint64_t coeff_baseline =
      d_multinomial_mod_p<M>(d_fact_M, d_fact_inv_M, vec, n_args, p, p_dash);

  // Step 5: Loop over rotations (david or jack)
  uint64_t ret = 0;
  for (size_t r_idx = 0; r_idx < M; ++r_idx) {
    if (vec[r_idx] == 0)
      continue;

    uint64_t coeff = d_mont_mul(coeff_baseline, d_nat_M[vec[r_idx]], p, p_dash);

    uint64_t f_n;
    if (is_jack_mode) {
      // Jack mode
      f_n = d_mont_mul(coeff, f_0, p, p_dash);
    } else {
      // David mode
      size_t idx = (2 * r_idx) % M;
      f_n = d_mont_mul(coeff,
                       d_mont_mul(f_0, d_ws_M[idx ? M - idx : 0], p, p_dash), p,
                       p_dash);
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

bool gpu_available(void) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  // Print GPU info on first call
  static bool printed = false;
  if (!printed) {
    fprintf(stderr, "Found %d GPU%s:\n", device_count,
            device_count > 1 ? "s" : "");
    for (int i = 0; i < device_count; i++) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      fprintf(stderr,
              "  GPU %d: %s (SM count: %d, Max threads per block: %d)\n", i,
              prop.name, prop.multiProcessorCount, prop.maxThreadsPerBlock);
    }
    printed = true;
  }

  return true;
}

int gpu_device_count(void) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess)
    return 0;
  return device_count;
}

struct vec_batch_t {
  // Host data (pinned for async transfers)
  uint64_t *h_vecs;    // max_vecs * m
  uint64_t *h_results; // max_vecs

  // Device data - matrix computation
  uint64_t *d_vecs;
  uint64_t *d_jk_prod_M;
  uint64_t *d_nat_M;
  uint64_t *d_nat_inv_M;

  // Device data - full computation lookup tables
  uint64_t *d_ws_M;                // m
  uint64_t *d_jk_sums_pow_lower_M; // POW_CACHE_DIVISOR * m_half
  uint64_t *d_jk_sums_pow_upper_M; // POW_CACHE_DIVISOR * m_half
  uint64_t *d_rs;                  // n_rs
  uint64_t *d_fact_M;              // n+1
  uint64_t *d_fact_inv_M;          // n+1

  uint64_t *d_results;

  // CUDA streams for pipelining
  cudaStream_t stream;

  // Batch parameters
  size_t max_vecs;
  size_t count;
  uint64_t n;
  uint64_t n_args;
  uint64_t m;
  size_t n_rs;
  bool is_jack_mode;

  // Montgomery parameters
  uint64_t p, p_dash, r, r3;
};

vec_batch_t *vec_batch_new(
    size_t max_vecs, uint64_t n, uint64_t n_args, uint64_t m, uint64_t p,
    uint64_t p_dash, uint64_t r, uint64_t r3, const uint64_t *jk_prod_M,
    const uint64_t *nat_M, const uint64_t *nat_inv_M, const uint64_t *ws_M,
    const uint64_t *jk_sums_pow_lower_M, const uint64_t *jk_sums_pow_upper_M,
    const uint64_t *rs, const uint64_t *fact_M, const uint64_t *fact_inv_M,
    size_t m_half, size_t n_rs, bool is_jack_mode) {
  vec_batch_t *batch = (vec_batch_t *)malloc(sizeof(vec_batch_t));
  assert(batch);

  batch->max_vecs = max_vecs;
  batch->count = 0;
  batch->n = n;
  batch->n_args = n_args;
  batch->m = m;
  batch->n_rs = n_rs;
  batch->p = p;
  batch->p_dash = p_dash;
  batch->r = r;
  batch->r3 = r3;
  batch->is_jack_mode = is_jack_mode;

  // Allocate pinned host memory for faster async transfers
  CUDA_CHECK(cudaMallocHost(&batch->h_vecs, max_vecs * m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMallocHost(&batch->h_results, max_vecs * sizeof(uint64_t)));

  CUDA_CHECK(cudaStreamCreate(&batch->stream));

  // Allocate device memory - matrix computation
  CUDA_CHECK(cudaMalloc(&batch->d_vecs, max_vecs * m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_jk_prod_M, m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_nat_M, (n + 1) * sizeof(uint64_t)));
  if (!is_jack_mode) {
    CUDA_CHECK(cudaMalloc(&batch->d_nat_inv_M, (n + 1) * sizeof(uint64_t)));
  } else {
    batch->d_nat_inv_M = NULL;
  }

  // Allocate device memory - full computation lookup tables
  CUDA_CHECK(cudaMalloc(&batch->d_ws_M, m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_jk_sums_pow_lower_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_jk_sums_pow_upper_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_rs, n_rs * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_fact_M, (n + 1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_fact_inv_M, (n + 1) * sizeof(uint64_t)));

  CUDA_CHECK(cudaMalloc(&batch->d_results, max_vecs * sizeof(uint64_t)));

  // Copy constant data to device - matrix computation
  CUDA_CHECK(cudaMemcpy(batch->d_jk_prod_M, jk_prod_M, m * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_nat_M, nat_M, (n + 1) * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  if (!is_jack_mode) {
    CUDA_CHECK(cudaMemcpy(batch->d_nat_inv_M, nat_inv_M,
                          (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));
  }

  // Copy constant data to device - full computation lookup tables
  CUDA_CHECK(cudaMemcpy(batch->d_ws_M, ws_M, m * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_jk_sums_pow_lower_M, jk_sums_pow_lower_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_jk_sums_pow_upper_M, jk_sums_pow_upper_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_rs, rs, n_rs * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_fact_M, fact_M, (n + 1) * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(batch->d_fact_inv_M, fact_inv_M,
                        (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

  return batch;
}

size_t vec_batch_add(vec_batch_t *batch, const uint64_t *vec) {
  assert(batch->count < batch->max_vecs);
  size_t idx = batch->count++;
  memcpy(&batch->h_vecs[idx * batch->m], vec, batch->m * sizeof(uint64_t));
  return idx;
}

#define LAUNCH_KERNEL(M, stream, count)                                        \
  vec_full_kernel<M, (M + 1) / 2><<<num_blocks, block_size, 0, stream>>>(      \
      batch->d_vecs, (count), batch->n, batch->n_args, batch->is_jack_mode,    \
      batch->d_jk_prod_M, batch->d_nat_M, batch->d_nat_inv_M, batch->d_ws_M,   \
      batch->d_jk_sums_pow_lower_M, batch->d_jk_sums_pow_upper_M, batch->d_rs, \
      batch->d_fact_M, batch->d_fact_inv_M, batch->d_results, batch->p,        \
      batch->p_dash, batch->r, batch->r3)

// Launch async GPU compute (non-blocking)
void vec_batch_compute_async(vec_batch_t *batch, batch_cb_t done, void *ud) {
  if (batch->count == 0)
    return;

  int block_size = 256;

  cudaStream_t stream = batch->stream;

  // Async copy H2D for this sub-batch
  CUDA_CHECK(cudaMemcpyAsync(batch->d_vecs, batch->h_vecs,
                             batch->count * batch->m * sizeof(uint64_t),
                             cudaMemcpyHostToDevice, stream));

  int num_blocks = (batch->count + block_size - 1) / block_size;

#define LK(n)                                                                  \
  case n:                                                                      \
    LAUNCH_KERNEL(n, stream, batch->count);                                    \
    break;

  switch (batch->m) {
    LK(3);
    LK(5);
    LK(7);
    LK(9);
    LK(11);
    LK(13);
    LK(15);
    LK(17);
    LK(19);
    LK(21);
    LK(23);
    LK(25);
  default:
    printf("\nm=%lu\n", batch->m);
    assert("unsupported m" && 0);
  }
#undef LK

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(batch->h_results, batch->d_results,
                             batch->count * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaLaunchHostFunc(batch->stream, done, ud));
}

// Wait for async compute to complete
void vec_batch_wait(vec_batch_t *batch) {
  CUDA_CHECK(cudaStreamSynchronize(batch->stream));
}

uint64_t vec_batch_get(const vec_batch_t *batch, size_t idx) {
  assert(idx < batch->count);
  return batch->h_results[idx];
}

void vec_batch_clear(vec_batch_t *batch) { batch->count = 0; }

void vec_batch_free(vec_batch_t *batch) {
  if (!batch)
    return;

  CUDA_CHECK(cudaStreamDestroy(batch->stream));

  // Free pinned host memory
  CUDA_CHECK(cudaFreeHost(batch->h_vecs));
  CUDA_CHECK(cudaFreeHost(batch->h_results));

  // Free device memory - matrix computation
  CUDA_CHECK(cudaFree(batch->d_vecs));
  CUDA_CHECK(cudaFree(batch->d_jk_prod_M));
  CUDA_CHECK(cudaFree(batch->d_nat_M));
  if (batch->d_nat_inv_M)
    CUDA_CHECK(cudaFree(batch->d_nat_inv_M));

  // Free device memory - full computation lookup tables
  CUDA_CHECK(cudaFree(batch->d_ws_M));
  CUDA_CHECK(cudaFree(batch->d_jk_sums_pow_lower_M));
  CUDA_CHECK(cudaFree(batch->d_jk_sums_pow_upper_M));
  CUDA_CHECK(cudaFree(batch->d_rs));
  CUDA_CHECK(cudaFree(batch->d_fact_M));
  CUDA_CHECK(cudaFree(batch->d_fact_inv_M));

  CUDA_CHECK(cudaFree(batch->d_results));

  free(batch);
}
