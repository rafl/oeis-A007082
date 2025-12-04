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

// Device-side context struct - passed to kernel as single pointer
struct gpu_kernel_ctx_t {
  // Device pointers to lookup tables
  fld_t *d_jk_prod_M;
  fld_t *d_nat_M;
  fld_t *d_nat_inv_M;
  fld_t *d_ws_M;
  fld_t *d_jk_sums_pow_lower_M;
  fld_t *d_jk_sums_pow_upper_M;
  fld_t *d_rs;
  fld_t *d_fact_M;
  fld_t *d_fact_inv_M;

  // Parameters
  uint64_t n;
  uint64_t n_args;
  fld_t p, p_dash, r, r3;
  bool is_jack_mode;
};

// Helper function for jk_pos on device
template <size_t M> __device__ inline size_t d_jk_pos(size_t j, size_t k) {
  int result = k - j;
  return result >= 0 ? (size_t)result : result + M;
}

__device__ inline fld_t d_add_mod(fld_t x, fld_t y, fld_t p) {
  x += y;
  sfld_t maybe = x - p;
  return maybe < 0 ? x : (fld_t)maybe;
}

// Device-side Montgomery arithmetic helpers
__device__ inline fld_t d_mont_mul(fld_t a, fld_t b, fld_t p, fld_t p_dash) {
  dfld_t t = (dfld_t)a * b;
  fld_t m = (fld_t)t * p_dash;
  dfld_t u = t + (dfld_t)m * p;
  fld_t res = u >> FLD_BITS;
  sfld_t maybe = res - p;
  return maybe < 0 ? res : (fld_t)maybe;
}

__device__ inline fld_t d_mont_pow(fld_t b, mss_el_t e, fld_t acc, fld_t p,
                                   fld_t p_dash) {
  while (e) {
    if (e & 1)
      acc = d_mont_mul(acc, b, p, p_dash);
    b = d_mont_mul(b, b, p, p_dash);
    e >>= 1;
  }
  return acc;
}

__device__ inline fld_t d_extended_euclidean(fld_t a, fld_t b) {
  fld_t r0 = a;
  fld_t r1 = b;
  fld_t s0 = 1;
  fld_t s1 = 0;
  fld_t spare;
  size_t n = 0;
  while (r1) {
    fld_t q = r0 / r1;
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

__device__ inline fld_t d_mont_inv(fld_t x, fld_t r3, fld_t p, fld_t p_dash) {
  fld_t inv = d_extended_euclidean(x, p);
  return d_mont_mul(r3, inv, p, p_dash);
}

__device__ inline fld_t d_mont_mul_sub(fld_t a1, fld_t b1, fld_t a2, fld_t b2,
                                       fld_t p, fld_t p_dash) {
  dfld_t t1 = (dfld_t)a1 * b1;
  dfld_t t2 = (dfld_t)a2 * b2;
  dfld_t t = t1 + ((dfld_t)p << FLD_BITS) - t2;
  fld_t m = (fld_t)t * p_dash;
  fld_t u = (t + (dfld_t)m * p) >> FLD_BITS;
  if (u >= p)
    u -= p;
  if (u >= p)
    u -= p;
  return u;
}

// Device-side fast_pow_2 using rs cache
__device__ inline fld_t d_fast_pow_2(const fld_t *d_rs, uint64_t pow, fld_t p,
                                     fld_t p_dash) {
  uint64_t r_pow = pow / FLD_BITS;
  uint64_t remain = pow % FLD_BITS;
  uint64_t pow2 = 1UL << remain;
  // TODO: reduce?
  return d_mont_mul(pow2, d_rs[r_pow + 2], p, p_dash);
}

// Device-side jk_sums_pow using split power cache
template <size_t M_HALF>
__device__ inline fld_t d_jk_sums_pow(const fld_t *d_jk_sums_pow_upper_M,
                                      const fld_t *d_jk_sums_pow_lower_M,
                                      fld_t diff, uint64_t pow, fld_t p,
                                      fld_t p_dash) {
  uint64_t upper_index = pow >> POW_CACHE_SPLIT;
  uint64_t lower_index = pow & (POW_CACHE_DIVISOR - 1);

  uint64_t upper_index_full = upper_index * M_HALF + diff;
  uint64_t lower_index_full = lower_index * M_HALF + diff;

  return d_mont_mul(d_jk_sums_pow_upper_M[upper_index_full],
                    d_jk_sums_pow_lower_M[lower_index_full], p, p_dash);
}

// Device-side multinomial coefficient computation
template <size_t M>
__device__ inline fld_t
d_multinomial_mod_p(const fld_t *d_fact_M, const fld_t *d_fact_inv_M,
                    const mss_el_t *vec, uint64_t n_args, fld_t p,
                    fld_t p_dash) {
  fld_t coeff = d_fact_M[n_args - 1];
  for (size_t i = 0; i < M; ++i) {
    coeff = d_mont_mul(coeff, d_fact_inv_M[vec[i]], p, p_dash);
  }
  return coeff;
}

// Device-side f_fst_trm computation
template <size_t M, size_t M_HALF>
__device__ inline fld_t d_f_fst_trm(const mss_el_t *vec, const fld_t *d_rs,
                                    const fld_t *d_jk_sums_pow_upper_M,
                                    const fld_t *d_jk_sums_pow_lower_M, fld_t p,
                                    fld_t p_dash) {
  uint16_t e = 0;
  // assert(m <= M);
  uint16_t pows[M];
  for (size_t i = 0; i < M; ++i) {
    pows[i] = 0;
  }

  for (size_t a = 0; a < M; ++a) {
    uint16_t ca = vec[a];
    if (!ca)
      continue;

    e += (ca * (ca - 1));

    for (size_t b = a + 1; b < M; ++b) {
      uint16_t cb = vec[b];
      uint16_t diff = b - a;
      pows[diff] += ca * cb;
    }
  }

  fld_t acc = d_fast_pow_2(d_rs, e / 2, p, p_dash);

  for (size_t i = 1; i < M_HALF; i++) {
    fld_t pow_val =
        d_jk_sums_pow<M_HALF>(d_jk_sums_pow_upper_M, d_jk_sums_pow_lower_M, i,
                              pows[i] + pows[M - i], p, p_dash);
    acc = d_mont_mul(acc, pow_val, p, p_dash);
  }

  return acc;
}

// Build matrix for f_snd_trm on GPU, return dimension and prod_M
template <size_t M>
__device__ size_t d_f_snd_trm_build_matrix(const mss_el_t *c,
                                           const fld_t *jk_prod_M,
                                           const fld_t *nat_M,
                                           const fld_t *nat_inv_M, fld_t *A,
                                           fld_t *prod_M_out, fld_t p,
                                           fld_t p_dash, fld_t r) {
  static_assert((uint8_t)-1 > M);
  uint8_t typ[M];
  size_t r_cnt = 0;
  for (size_t i = 0; i < M; ++i) {
    if (c[i]) {
      typ[r_cnt] = i;
      ++r_cnt;
    }
  }

  fld_t prod_M = r;

  for (size_t a = 0; a < r_cnt; ++a) {
    uint8_t i = typ[a];
    if (c[i] == 1)
      continue;

    fld_t sum = 0;
    for (size_t b = 0; b < r_cnt; ++b) {
      uint8_t j = typ[b];
      fld_t W = jk_prod_M[d_jk_pos<M>(i, j)];
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
    uint8_t i = typ[a];
    fld_t W_del = jk_prod_M[M - i];
    fld_t diag = d_mont_mul(nat_M[c[0]], W_del, p, p_dash);

    for (size_t b = 1; b < r_cnt; ++b) {
      uint8_t j = typ[b];
      if (j == i)
        continue;

      fld_t W = jk_prod_M[d_jk_pos<M>(i, j)];
      fld_t v = d_mont_mul(nat_M[c[j]], W, p, p_dash);
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
__device__ size_t d_jack_snd_trm_build_matrix(const mss_el_t *c,
                                              const fld_t *jk_prod_M,
                                              const fld_t *nat_M, fld_t *A,
                                              fld_t *prod_M_out, fld_t p,
                                              fld_t p_dash, fld_t r) {
  static_assert((uint8_t)-1 > M);
  uint8_t typ[M];
  size_t dim = 0;
  for (size_t i = 0; i < M; ++i) {
    if (c[i]) {
      typ[dim] = i;
      ++dim;
    }
  }

  fld_t prod_M = r;

  for (size_t a = 0; a < dim; ++a) {
    uint8_t i = typ[a];
    fld_t sum = r;
    for (size_t b = 0; b < dim; ++b) {
      uint8_t j = typ[b];
      fld_t w = jk_prod_M[d_jk_pos<M>(i, j)];
      sum = d_add_mod(sum, d_mont_mul(nat_M[c[j]], w, p, p_dash), p);
    }
    prod_M = d_mont_pow(sum, c[i] - 1, prod_M, p, p_dash);
  }

  if (dim <= 1) {
    *prod_M_out = prod_M;
    return 0;
  }

  for (size_t a = 0; a < dim; ++a) {
    uint8_t i = typ[a];
    fld_t diag = r;

    for (size_t b = 0; b < dim; ++b) {
      uint8_t j = typ[b];
      if (j == i)
        continue;

      fld_t w = jk_prod_M[d_jk_pos<M>(i, j)];
      fld_t v = d_mont_mul(nat_M[c[j]], w, p, p_dash);
      A[(a)*dim + (b)] = p - v;
      diag = d_add_mod(diag, v, p);
    }

    A[(a)*dim + (a)] = diag;
  }

  *prod_M_out = prod_M;
  return dim;
}

// Compute f_snd_trm for David mode: build matrix and compute determinant
template <size_t M, size_t DIM>
__device__ fld_t d_compute_f_snd_trm_david(const mss_el_t *vec, const fld_t *d_jk_prod_M,
                                           const fld_t *d_nat_M, const fld_t *d_nat_inv_M,
                                           fld_t p, fld_t p_dash, fld_t r, fld_t r3) {
  fld_t prod_M;

  if constexpr (DIM == 0) {
    // DIM=0: no matrix needed
    d_f_snd_trm_build_matrix<M>(vec, d_jk_prod_M, d_nat_M, d_nat_inv_M, NULL,
                                 &prod_M, p, p_dash, r);
    return prod_M;
  } else {
    // DIM > 0: build matrix and compute determinant
    fld_t A[DIM * DIM];
    size_t dim = DIM;

    d_f_snd_trm_build_matrix<M>(vec, d_jk_prod_M, d_nat_M, d_nat_inv_M, A,
                                       &prod_M, p, p_dash, r);

    // Compute determinant via Gaussian elimination
    fld_t det = r, scaling_factor = r;

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
          fld_t tmp = A[k * dim + j];
          A[k * dim + j] = A[pivot_i * dim + j];
          A[pivot_i * dim + j] = tmp;
        }
        det = p - det;
      }

      fld_t pivot = A[k * dim + k];
      det = d_mont_mul(det, A[k * dim + k], p, p_dash);

      // Elimination
      for (size_t i = k + 1; i < dim; ++i) {
        scaling_factor = d_mont_mul(scaling_factor, pivot, p, p_dash);
        fld_t multiplier = A[i * dim + k];
        for (size_t j = k; j < dim; ++j) {
          A[i * dim + j] = d_mont_mul_sub(A[i * dim + j], pivot, A[k * dim + j],
                                          multiplier, p, p_dash);
        }
      }
    }

    det = d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p, p_dash);

    return d_mont_mul(prod_M, det, p, p_dash);
  }
}

// Compute f_snd_trm for Jack mode: build matrix and compute determinant
template <size_t M, size_t DIM>
__device__ fld_t d_compute_f_snd_trm_jack(const mss_el_t *vec, const fld_t *d_jk_prod_M,
                                          const fld_t *d_nat_M,
                                          fld_t p, fld_t p_dash, fld_t r, fld_t r3) {
  fld_t prod_M;

  if constexpr (DIM <= 1) {
    // DIM <= 1: no matrix needed, builder returns 0
    d_jack_snd_trm_build_matrix<M>(vec, d_jk_prod_M, d_nat_M, NULL, &prod_M,
                                    p, p_dash, r);
    return prod_M;
  } else {
    // DIM > 1: build matrix and compute determinant
    fld_t A[DIM * DIM];

    d_jack_snd_trm_build_matrix<M>(vec, d_jk_prod_M, d_nat_M, A, &prod_M,
                                    p, p_dash, r);

    // Compute determinant via Gaussian elimination
    fld_t det = r, scaling_factor = r;

    for (size_t k = 0; k < DIM; ++k) {
      // Find pivot
      size_t pivot_i = k;
      while (pivot_i < DIM && A[pivot_i * DIM + k] == 0)
        ++pivot_i;

      if (pivot_i == DIM) {
        det = 0;
        break;
      }

      // Swap rows if needed
      if (pivot_i != k) {
        for (size_t j = 0; j < DIM; ++j) {
          fld_t tmp = A[k * DIM + j];
          A[k * DIM + j] = A[pivot_i * DIM + j];
          A[pivot_i * DIM + j] = tmp;
        }
        det = p - det;
      }

      fld_t pivot = A[k * DIM + k];
      det = d_mont_mul(det, A[k * DIM + k], p, p_dash);

      // Elimination
      for (size_t i = k + 1; i < DIM; ++i) {
        scaling_factor = d_mont_mul(scaling_factor, pivot, p, p_dash);
        fld_t multiplier = A[i * DIM + k];
        for (size_t j = k; j < DIM; ++j) {
          A[i * DIM + j] = d_mont_mul_sub(A[i * DIM + j], pivot, A[k * DIM + j],
                                          multiplier, p, p_dash);
        }
      }
    }

    det = d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p, p_dash);

    return d_mont_mul(prod_M, det, p, p_dash);
  }
}

// Comprehensive kernel: computes full david() or jack() result on GPU
// Each thread processes one coefficient vector and produces final result
template <size_t M, size_t M_HALF, size_t DIM>
__global__ void vec_full_kernel(const mss_el_t *vecs, size_t n_vecs,
                                const gpu_kernel_ctx_t *ctx, fld_t *results) {
  size_t vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_idx >= n_vecs)
    return;

  // Load context values into registers
  const fld_t p = ctx->p;
  const fld_t p_dash = ctx->p_dash;
  const fld_t r = ctx->r;
  const fld_t r3 = ctx->r3;
  const uint64_t n = ctx->n;
  const uint64_t n_args = ctx->n_args;
  const bool is_jack_mode = ctx->is_jack_mode;

  const fld_t *d_jk_prod_M = ctx->d_jk_prod_M;
  const fld_t *d_nat_M = ctx->d_nat_M;
  const fld_t *d_nat_inv_M = ctx->d_nat_inv_M;
  const fld_t *d_ws_M = ctx->d_ws_M;
  const fld_t *d_jk_sums_pow_lower_M = ctx->d_jk_sums_pow_lower_M;
  const fld_t *d_jk_sums_pow_upper_M = ctx->d_jk_sums_pow_upper_M;
  const fld_t *d_rs = ctx->d_rs;
  const fld_t *d_fact_M = ctx->d_fact_M;
  const fld_t *d_fact_inv_M = ctx->d_fact_inv_M;

  const mss_el_t *vec = &vecs[vec_idx * M];

  // Step 1: Compute f_fst_trm
  fld_t f_fst_result = d_f_fst_trm<M, M_HALF>(vec, d_rs, d_jk_sums_pow_upper_M,
                                              d_jk_sums_pow_lower_M, p, p_dash);

  // Step 2: Compute f_snd_trm (build matrix + compute determinant)
  fld_t f_snd_result = is_jack_mode
                           ? d_compute_f_snd_trm_jack<M, DIM + 1>(vec, d_jk_prod_M, d_nat_M,
                                                                   p, p_dash, r, r3)
                           : d_compute_f_snd_trm_david<M, DIM>(vec, d_jk_prod_M, d_nat_M,
                                                                d_nat_inv_M, p, p_dash, r, r3);

  // Step 3: Multiply f_fst_result and f_snd_result to get f_0
  fld_t f_0 = d_mont_mul(f_fst_result, f_snd_result, p, p_dash);

  // Step 4: Compute multinomial coefficient
  fld_t coeff_baseline =
      d_multinomial_mod_p<M>(d_fact_M, d_fact_inv_M, vec, n_args, p, p_dash);

  // Step 5: Loop over rotations (david or jack)
  fld_t ret = 0;
  for (size_t r_idx = 0; r_idx < M; ++r_idx) {
    if (vec[r_idx] == 0)
      continue;

    fld_t coeff = d_mont_mul(coeff_baseline, d_nat_M[vec[r_idx]], p, p_dash);

    fld_t f_n;
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

// Shared context for constant data (one per worker)
struct gpu_context_t {
  // Device data - constant lookup tables
  fld_t *d_jk_prod_M;           // m
  fld_t *d_nat_M;               // n+1
  fld_t *d_nat_inv_M;           // n+1 (NULL for jack mode)
  fld_t *d_ws_M;                // m
  fld_t *d_jk_sums_pow_lower_M; // POW_CACHE_DIVISOR * m_half
  fld_t *d_jk_sums_pow_upper_M; // POW_CACHE_DIVISOR * m_half
  fld_t *d_rs;                  // n_rs
  fld_t *d_fact_M;              // n+1
  fld_t *d_fact_inv_M;          // n+1

  // Device-side kernel context (single struct with all pointers/params)
  gpu_kernel_ctx_t *d_ctx;

  // Context parameters
  uint64_t n;
  uint64_t n_args;
  uint64_t m;
  size_t m_half;
  size_t n_rs;
  bool is_jack_mode;

  // Montgomery parameters
  fld_t p, p_dash, r, r3;
};

// Per-batch data (multiple per worker for pipelining)
struct vec_batch_t {
  // Host data (pinned for async transfers)
  mss_el_t *h_vecs; // max_vecs * m
  fld_t *h_results; // max_vecs

  // Device data - per-batch only
  mss_el_t *d_vecs;
  fld_t *d_results;

  // CUDA stream for pipelining
  cudaStream_t stream;

  // Batch parameters
  size_t max_vecs;
  size_t count;

  // Pointer to shared context
  gpu_context_t *ctx;
};

gpu_context_t *gpu_context_new(
    uint64_t n, uint64_t n_args, uint64_t m, fld_t p, fld_t p_dash, fld_t r,
    fld_t r3, const fld_t *jk_prod_M, const fld_t *nat_M,
    const fld_t *nat_inv_M, const fld_t *ws_M, const fld_t *jk_sums_pow_lower_M,
    const fld_t *jk_sums_pow_upper_M, const fld_t *rs, const fld_t *fact_M,
    const fld_t *fact_inv_M, size_t m_half, size_t n_rs, bool is_jack_mode) {
  gpu_context_t *ctx = (gpu_context_t *)malloc(sizeof(gpu_context_t));
  assert(ctx);

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

  // Allocate device memory for constant lookup tables
  CUDA_CHECK(cudaMalloc(&ctx->d_jk_prod_M, m * sizeof(fld_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_nat_M, (n + 1) * sizeof(fld_t)));
  if (!is_jack_mode) {
    CUDA_CHECK(cudaMalloc(&ctx->d_nat_inv_M, (n + 1) * sizeof(fld_t)));
  } else {
    ctx->d_nat_inv_M = NULL;
  }
  CUDA_CHECK(cudaMalloc(&ctx->d_ws_M, m * sizeof(fld_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_jk_sums_pow_lower_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(fld_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_jk_sums_pow_upper_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(fld_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_rs, n_rs * sizeof(fld_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_fact_M, (n + 1) * sizeof(fld_t)));
  CUDA_CHECK(cudaMalloc(&ctx->d_fact_inv_M, (n + 1) * sizeof(fld_t)));

  // Copy constant data to device
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_prod_M, jk_prod_M, m * sizeof(fld_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_nat_M, nat_M, (n + 1) * sizeof(fld_t),
                        cudaMemcpyHostToDevice));
  if (!is_jack_mode) {
    CUDA_CHECK(cudaMemcpy(ctx->d_nat_inv_M, nat_inv_M, (n + 1) * sizeof(fld_t),
                          cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(
      cudaMemcpy(ctx->d_ws_M, ws_M, m * sizeof(fld_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_sums_pow_lower_M, jk_sums_pow_lower_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(fld_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_sums_pow_upper_M, jk_sums_pow_upper_M,
                        POW_CACHE_DIVISOR * m_half * sizeof(fld_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(ctx->d_rs, rs, n_rs * sizeof(fld_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_fact_M, fact_M, (n + 1) * sizeof(fld_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ctx->d_fact_inv_M, fact_inv_M, (n + 1) * sizeof(fld_t),
                        cudaMemcpyHostToDevice));

  // Create device-side kernel context struct
  gpu_kernel_ctx_t h_ctx = {
      .d_jk_prod_M = ctx->d_jk_prod_M,
      .d_nat_M = ctx->d_nat_M,
      .d_nat_inv_M = ctx->d_nat_inv_M,
      .d_ws_M = ctx->d_ws_M,
      .d_jk_sums_pow_lower_M = ctx->d_jk_sums_pow_lower_M,
      .d_jk_sums_pow_upper_M = ctx->d_jk_sums_pow_upper_M,
      .d_rs = ctx->d_rs,
      .d_fact_M = ctx->d_fact_M,
      .d_fact_inv_M = ctx->d_fact_inv_M,
      .n = n,
      .n_args = n_args,
      .p = p,
      .p_dash = p_dash,
      .r = r,
      .r3 = r3,
      .is_jack_mode = is_jack_mode,
  };
  CUDA_CHECK(cudaMalloc(&ctx->d_ctx, sizeof(gpu_kernel_ctx_t)));
  CUDA_CHECK(cudaMemcpy(ctx->d_ctx, &h_ctx, sizeof(gpu_kernel_ctx_t),
                        cudaMemcpyHostToDevice));

  return ctx;
}

void gpu_context_free(gpu_context_t *ctx) {
  if (!ctx)
    return;

  CUDA_CHECK(cudaFree(ctx->d_jk_prod_M));
  CUDA_CHECK(cudaFree(ctx->d_nat_M));
  if (ctx->d_nat_inv_M)
    CUDA_CHECK(cudaFree(ctx->d_nat_inv_M));
  CUDA_CHECK(cudaFree(ctx->d_ws_M));
  CUDA_CHECK(cudaFree(ctx->d_jk_sums_pow_lower_M));
  CUDA_CHECK(cudaFree(ctx->d_jk_sums_pow_upper_M));
  CUDA_CHECK(cudaFree(ctx->d_rs));
  CUDA_CHECK(cudaFree(ctx->d_fact_M));
  CUDA_CHECK(cudaFree(ctx->d_fact_inv_M));
  CUDA_CHECK(cudaFree(ctx->d_ctx));

  free(ctx);
}

vec_batch_t *vec_batch_new(gpu_context_t *ctx, size_t max_vecs) {
  vec_batch_t *batch = (vec_batch_t *)malloc(sizeof(vec_batch_t));
  assert(batch);

  batch->ctx = ctx;
  batch->max_vecs = max_vecs;
  batch->count = 0;

  // Allocate pinned host memory for faster async transfers
  CUDA_CHECK(
      cudaMallocHost(&batch->h_vecs, max_vecs * ctx->m * sizeof(mss_el_t)));
  CUDA_CHECK(cudaMallocHost(&batch->h_results, max_vecs * sizeof(fld_t)));

  CUDA_CHECK(cudaStreamCreate(&batch->stream));

  // Allocate device memory - per-batch only
  CUDA_CHECK(cudaMalloc(&batch->d_vecs, max_vecs * ctx->m * sizeof(mss_el_t)));
  CUDA_CHECK(cudaMalloc(&batch->d_results, max_vecs * sizeof(fld_t)));

  return batch;
}

void vec_batch_add_bulk(vec_batch_t *batch, const mss_el_t *vecs,
                        size_t count) {
  assert(count <= batch->max_vecs);
  size_t m = batch->ctx->m;
  memcpy(batch->h_vecs, vecs, count * m * sizeof(mss_el_t));
  batch->count = count;
}

#define LAUNCH_KERNEL(M, DIM, stream, count)                                   \
  vec_full_kernel<M, (M + 1) / 2, DIM><<<num_blocks, block_size, 0, stream>>>( \
      batch->d_vecs, (count), batch->ctx->d_ctx, batch->d_results)

// Launch async GPU compute (non-blocking)
void vec_batch_compute_async(vec_batch_t *batch, uint8_t vec_class,
                             batch_cb_t done, void *ud) {
  if (batch->count == 0)
    return;

  int block_size = 256;
  size_t m = batch->ctx->m;

  cudaStream_t stream = batch->stream;

  // Async copy H2D for this sub-batch
  CUDA_CHECK(cudaMemcpyAsync(batch->d_vecs, batch->h_vecs,
                             batch->count * m * sizeof(mss_el_t),
                             cudaMemcpyHostToDevice, stream));

  int num_blocks = (batch->count + block_size - 1) / block_size;

#define LK_DIM(M, DIM)                                                         \
  case DIM:                                                                    \
    LAUNCH_KERNEL(M, DIM, stream, batch->count);                               \
    break;

#define LK(n)                                                                  \
  case n: {                                                                    \
    uint8_t dim = vec_class-1;                                               \
    switch (dim) {                                                             \
      LK_DIM(n, 0);                                                            \
      LK_DIM(n, 1);                                                            \
      LK_DIM(n, 2);                                                            \
      LK_DIM(n, 3);                                                            \
      LK_DIM(n, 4);                                                            \
      LK_DIM(n, 5);                                                            \
      LK_DIM(n, 6);                                                            \
      LK_DIM(n, 7);                                                            \
      LK_DIM(n, 8);                                                            \
      LK_DIM(n, 9);                                                            \
      LK_DIM(n, 10);                                                           \
      LK_DIM(n, 11);                                                           \
      LK_DIM(n, 12);                                                           \
      LK_DIM(n, 13);                                                           \
      LK_DIM(n, 14);                                                           \
      LK_DIM(n, 15);                                                           \
      LK_DIM(n, 16);                                                           \
      LK_DIM(n, 17);                                                           \
      LK_DIM(n, 18);                                                           \
      LK_DIM(n, 19);                                                           \
      LK_DIM(n, 20);                                                           \
      LK_DIM(n, 21);                                                           \
    default:                                                                   \
      printf("\ndim=%u\n", dim);                                               \
      assert("unsupported dim" && 0);                                          \
    }                                                                          \
    break;                                                                     \
  }

  switch (m) {
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
    printf("\nm=%lu\n", m);
    assert("unsupported m" && 0);
  }
#undef LK
#undef LK_DIM

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(batch->h_results, batch->d_results,
                             batch->count * sizeof(fld_t),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaLaunchHostFunc(batch->stream, done, ud));
}

fld_t vec_batch_get(const vec_batch_t *batch, size_t idx) {
  assert(idx < batch->count);
  return batch->h_results[idx];
}

void vec_batch_free(vec_batch_t *batch) {
  if (!batch)
    return;

  CUDA_CHECK(cudaStreamDestroy(batch->stream));

  // Free pinned host memory
  CUDA_CHECK(cudaFreeHost(batch->h_vecs));
  CUDA_CHECK(cudaFreeHost(batch->h_results));

  // Free device memory - per-batch only
  CUDA_CHECK(cudaFree(batch->d_vecs));
  CUDA_CHECK(cudaFree(batch->d_results));

  free(batch);
}
