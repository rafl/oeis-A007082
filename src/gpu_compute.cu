#include "gpu_compute.h"
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

// Device-side Montgomery arithmetic
__device__ inline uint64_t d_add_mod(uint64_t x, uint64_t y, uint64_t p) {
  x += y;
  int64_t maybe = x - p;
  return maybe < 0 ? x : (uint64_t)maybe;
}

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

__device__ inline uint64_t d_inv64_u64(uint64_t p) {
  uint64_t x = 1;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  x *= 2 - p * x;
  return x;
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

__device__ inline size_t d_jk_pos(size_t j, size_t k, uint64_t m) {
  int64_t result = k - j;
  return result >= 0 ? (uint64_t)result : result + m;
}

// GPU context structure
struct gpu_ctx_t {
  uint64_t n, n_args, m, p, p_dash, r, r2, r3;
  uint64_t *d_ws_M;
  uint64_t *d_jk_prod_M;
  uint64_t *d_jk_sums_M;
  uint64_t *d_nat_M;
  uint64_t *d_nat_inv_M;
  uint64_t *d_fact_M;
  uint64_t *d_fact_inv_M;
  uint64_t *d_rs;
  size_t n_rs;
  bool jack_mode;
  uint64_t m_half;
};

// Multinomial coefficient calculation on device
__device__ uint64_t d_multinomial(const size_t *ms, size_t len,
                                  const uint64_t *fact_M,
                                  const uint64_t *fact_inv_M, uint64_t n_args,
                                  uint64_t p, uint64_t p_dash) {
  uint64_t coeff = fact_M[n_args - 1];
  for (size_t i = 0; i < len; ++i)
    coeff = d_mont_mul(coeff, fact_inv_M[ms[i]], p, p_dash);
  return coeff;
}

// Fast power of 2 on device
__device__ uint64_t d_fast_pow_2(uint64_t pow, const uint64_t *rs, uint64_t p,
                                 uint64_t p_dash) {
  uint64_t r_pow = pow / 64;
  uint64_t remain = pow % 64;
  uint64_t pow2 = 1UL << remain;
  return d_mont_mul(pow2, rs[r_pow + 2], p, p_dash);
}

// Determinant calculation on device
__device__ uint64_t d_det_mod_p(uint64_t *A, size_t dim, uint64_t r,
                                uint64_t r3, uint64_t p, uint64_t p_dash) {
  uint64_t det = r, scaling_factor = r;

  for (size_t k = 0; k < dim; ++k) {
    size_t pivot_i = k;
    while (pivot_i < dim && A[pivot_i * dim + k] == 0)
      ++pivot_i;

    if (pivot_i == dim)
      return 0;

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

    for (size_t i = k + 1; i < dim; ++i) {
      scaling_factor = d_mont_mul(scaling_factor, pivot, p, p_dash);
      uint64_t multiplier = A[i * dim + k];
      for (size_t j = k; j < dim; ++j)
        A[i * dim + j] = d_mont_mul_sub(A[i * dim + j], pivot, A[k * dim + j],
                                        multiplier, p, p_dash);
    }
  }

  return d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p, p_dash);
}

// Maximum m value we support (for stack allocation safety)
#define MAX_M 64

// f_fst_trm calculation
__device__ uint64_t d_f_fst_trm(const size_t *c, uint64_t m, uint64_t m_half,
                                const uint64_t *jk_sums_M, const uint64_t *rs,
                                uint64_t p, uint64_t p_dash) {
  if (m > MAX_M) return 0; // Safety check

  uint64_t pows[MAX_M];
  for (size_t i = 0; i < m; ++i)
    pows[i] = 0;

  uint64_t e = 0;
  for (size_t a = 0; a < m; ++a) {
    uint64_t ca = c[a];
    if (!ca)
      continue;
    e += (ca * (ca - 1));
    for (size_t b = a + 1; b < m; ++b) {
      uint64_t cb = c[b];
      uint64_t diff = b - a;
      pows[diff] += ca * cb;
    }
  }

  uint64_t acc = d_fast_pow_2(e / 2, rs, p, p_dash);

  for (size_t i = 1; i < m_half; i++) {
    uint64_t total_pow = pows[i] + pows[m - i];
    if (total_pow > 0) {
      uint64_t pow_val = d_mont_pow(jk_sums_M[i], total_pow, 1, p, p_dash);
      uint64_t r_val = rs[2];
      pow_val = d_mont_mul(pow_val, r_val, p, p_dash);
      acc = d_mont_mul(acc, pow_val, p, p_dash);
    }
  }

  return acc;
}

// f_snd_trm calculation
__device__ uint64_t d_f_snd_trm(const size_t *c, uint64_t m,
                                const uint64_t *jk_prod_M,
                                const uint64_t *nat_M, const uint64_t *nat_inv_M,
                                uint64_t r, uint64_t r3, uint64_t p,
                                uint64_t p_dash) {
  if (m > MAX_M) return 0; // Safety check

  size_t typ[MAX_M];
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
  if (!dim)
    return prod_M;

  if (dim > MAX_M) return 0; // Safety check

  uint64_t A[MAX_M * MAX_M];
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

  return d_mont_mul(prod_M, d_det_mod_p(A, dim, r, r3, p, p_dash), p, p_dash);
}

// Full f function (david mode)
__device__ uint64_t d_f(const size_t *vec, uint64_t m, uint64_t m_half,
                        const uint64_t *jk_prod_M, const uint64_t *jk_sums_M,
                        const uint64_t *nat_M, const uint64_t *nat_inv_M,
                        const uint64_t *rs, uint64_t r, uint64_t r3, uint64_t p,
                        uint64_t p_dash) {
  uint64_t fst = d_f_fst_trm(vec, m, m_half, jk_sums_M, rs, p, p_dash);
  uint64_t snd =
      d_f_snd_trm(vec, m, jk_prod_M, nat_M, nat_inv_M, r, r3, p, p_dash);
  return d_mont_mul(fst, snd, p, p_dash);
}

// jack_snd_trm calculation
__device__ uint64_t d_jack_snd_trm(const size_t *c, uint64_t m,
                                   const uint64_t *jk_prod_M,
                                   const uint64_t *nat_M, uint64_t r,
                                   uint64_t r3, uint64_t p, uint64_t p_dash) {
  if (m > MAX_M) return 0; // Safety check

  size_t typ[MAX_M];
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

  if (r_cnt <= 1)
    return prod_M;

  if (r_cnt > MAX_M) return 0; // Safety check

  uint64_t A[MAX_M * MAX_M];
  for (size_t a = 0; a < r_cnt; ++a) {
    size_t i = typ[a];
    uint64_t diag = r;

    for (size_t b = 0; b < r_cnt; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      uint64_t w = jk_prod_M[d_jk_pos(i, j, m)];
      uint64_t v = d_mont_mul(nat_M[c[j]], w, p, p_dash);
      A[(a)*r_cnt + (b)] = p - v;
      diag = d_add_mod(diag, v, p);
    }

    A[(a)*r_cnt + (a)] = diag;
  }

  uint64_t ret = d_mont_mul(prod_M, d_det_mod_p(A, r_cnt, r, r3, p, p_dash), p,
                            p_dash);
  return ret;
}

// jack_offset function
__device__ uint64_t d_jack_offset(const size_t *vec, uint64_t m,
                                  uint64_t m_half, const uint64_t *jk_prod_M,
                                  const uint64_t *jk_sums_M,
                                  const uint64_t *nat_M, const uint64_t *rs,
                                  uint64_t r, uint64_t r3, uint64_t p,
                                  uint64_t p_dash) {
  uint64_t fst = d_f_fst_trm(vec, m, m_half, jk_sums_M, rs, p, p_dash);
  uint64_t snd = d_jack_snd_trm(vec, m, jk_prod_M, nat_M, r, r3, p, p_dash);
  return d_mont_mul(fst, snd, p, p_dash);
}

// jack function
__device__ uint64_t d_jack(const size_t *vec, uint64_t m, uint64_t m_half,
                           const uint64_t *jk_prod_M,
                           const uint64_t *jk_sums_M, const uint64_t *nat_M,
                           const uint64_t *fact_M, const uint64_t *fact_inv_M,
                           const uint64_t *rs, uint64_t n, uint64_t n_args,
                           uint64_t r, uint64_t r3, uint64_t p,
                           uint64_t p_dash) {
  uint64_t ret = 0;
  uint64_t f_0 =
      d_jack_offset(vec, m, m_half, jk_prod_M, jk_sums_M, nat_M, rs, r, r3, p,
                    p_dash);
  uint64_t coeff_baseline =
      d_multinomial(vec, m, fact_M, fact_inv_M, n_args, p, p_dash);

  for (size_t r_idx = 0; r_idx < m; ++r_idx) {
    if (vec[r_idx] == 0)
      continue;

    size_t coeff = d_mont_mul(coeff_baseline, nat_M[vec[r_idx]], p, p_dash);
    uint64_t f_n = d_mont_mul(coeff, f_0, p, p_dash);
    ret = d_add_mod(ret, f_n, p);
  }

  ret = d_mont_mul(ret, nat_M[n - 1], p, p_dash);
  return d_mont_mul(ret, nat_M[n - 1], p, p_dash);
}

// david function
__device__ uint64_t d_david(const size_t *vec, uint64_t m, uint64_t m_half,
                            const uint64_t *jk_prod_M,
                            const uint64_t *jk_sums_M, const uint64_t *nat_M,
                            const uint64_t *nat_inv_M, const uint64_t *fact_M,
                            const uint64_t *fact_inv_M, const uint64_t *ws_M,
                            const uint64_t *rs, uint64_t n_args, uint64_t r,
                            uint64_t r3, uint64_t p, uint64_t p_dash) {
  uint64_t ret = 0;
  uint64_t f_0 =
      d_f(vec, m, m_half, jk_prod_M, jk_sums_M, nat_M, nat_inv_M, rs, r, r3, p,
          p_dash);
  uint64_t coeff_baseline =
      d_multinomial(vec, m, fact_M, fact_inv_M, n_args, p, p_dash);

  for (size_t r_idx = 0; r_idx < m; ++r_idx) {
    if (vec[r_idx] == 0)
      continue;

    size_t coeff = d_mont_mul(coeff_baseline, nat_M[vec[r_idx]], p, p_dash);
    size_t idx = (2 * r_idx) % m;
    uint64_t f_n = d_mont_mul(
        coeff, d_mont_mul(f_0, ws_M[idx ? m - idx : 0], p, p_dash), p, p_dash);

    ret = d_add_mod(ret, f_n, p);
  }

  return ret;
}

// CUDA kernel to process batch of vectors
__global__ void compute_kernel(const size_t *vecs, size_t n_vecs, uint64_t *results,
                               uint64_t n, uint64_t n_args, uint64_t m,
                               uint64_t m_half, uint64_t p, uint64_t p_dash,
                               uint64_t r, uint64_t r2, uint64_t r3,
                               const uint64_t *ws_M, const uint64_t *jk_prod_M,
                               const uint64_t *jk_sums_M, const uint64_t *nat_M,
                               const uint64_t *nat_inv_M, const uint64_t *fact_M,
                               const uint64_t *fact_inv_M, const uint64_t *rs,
                               bool jack_mode) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vecs)
    return;

  const size_t *vec = &vecs[idx * m];
  uint64_t result;

  if (jack_mode) {
    result = d_jack(vec, m, m_half, jk_prod_M, jk_sums_M, nat_M, fact_M,
                    fact_inv_M, rs, n, n_args, r, r3, p, p_dash);
  } else {
    result = d_david(vec, m, m_half, jk_prod_M, jk_sums_M, nat_M, nat_inv_M,
                     fact_M, fact_inv_M, ws_M, rs, n_args, r, r3, p, p_dash);
  }

  results[idx] = result;
}

// Reduction kernel to sum results
__global__ void reduce_kernel(uint64_t *results, size_t n, uint64_t p) {
  extern __shared__ uint64_t sdata[];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? results[i] : 0;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = d_add_mod(sdata[tid], sdata[tid + s], p);
    }
    __syncthreads();
  }

  if (tid == 0)
    results[blockIdx.x] = sdata[0];
}

bool gpu_available(void) {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

gpu_ctx_t *gpu_ctx_new(uint64_t n, uint64_t n_args, uint64_t m, uint64_t p,
                       uint64_t w, bool jack_mode) {
  gpu_ctx_t *ctx = (gpu_ctx_t *)malloc(sizeof(gpu_ctx_t));
  assert(ctx);

  ctx->n = n;
  ctx->n_args = n_args;
  ctx->m = m;
  ctx->p = p;
  ctx->m_half = (m + 1) / 2;
  ctx->jack_mode = jack_mode;

  // Compute Montgomery parameters
  uint64_t p_inv = 1;
  for (int i = 0; i < 6; i++)
    p_inv *= 2 - p * p_inv;
  ctx->p_dash = (uint64_t)(-p_inv);

  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;
  ctx->r3 = (uint128_t)ctx->r2 * ctx->r % p;

  // Helper function for host Montgomery multiplication
  auto mont_mul_host = [](uint64_t a, uint64_t b, uint64_t p,
                          uint64_t p_dash) -> uint64_t {
    uint128_t t = (uint128_t)a * b;
    uint64_t m = (uint64_t)t * p_dash;
    uint128_t u = t + (uint128_t)m * p;
    uint64_t res = u >> 64;
    int64_t maybe = res - p;
    return maybe < 0 ? res : (uint64_t)maybe;
  };

  auto mont_inv_host = [&](uint64_t x, uint64_t r3, uint64_t p,
                           uint64_t p_dash) -> uint64_t {
    uint64_t r0 = x, r1 = p, s0 = 1, s1 = 0;
    size_t n = 0;
    while (r1) {
      uint64_t q = r0 / r1;
      uint64_t spare = r0 % r1;
      r0 = r1;
      r1 = spare;
      spare = s0 + q * s1;
      s0 = s1;
      s1 = spare;
      ++n;
    }
    if (n % 2)
      s0 = p - s0;
    return mont_mul_host(r3, s0, p, p_dash);
  };

  auto mont_pow_host = [&](uint64_t b, uint64_t e, uint64_t acc, uint64_t p,
                           uint64_t p_dash) -> uint64_t {
    while (e) {
      if (e & 1)
        acc = mont_mul_host(acc, b, p, p_dash);
      b = mont_mul_host(b, b, p, p_dash);
      e >>= 1;
    }
    return acc;
  };

  // Allocate and initialize roots of unity
  uint64_t *ws_M = (uint64_t *)malloc(m * sizeof(uint64_t));
  ws_M[0] = ctx->r;
  ws_M[1] = mont_mul_host(w, ctx->r2, p, ctx->p_dash);
  for (size_t i = 2; i < m; ++i)
    ws_M[i] = mont_mul_host(ws_M[i - 1], ws_M[1], p, ctx->p_dash);

  CUDA_CHECK(cudaMalloc(&ctx->d_ws_M, m * sizeof(uint64_t)));
  CUDA_CHECK(
      cudaMemcpy(ctx->d_ws_M, ws_M, m * sizeof(uint64_t), cudaMemcpyHostToDevice));
  free(ws_M);

  // Compute and upload jk_prod_M and jk_sums_M
  uint64_t *jk_pairs_M = (uint64_t *)malloc(m * m * sizeof(uint64_t));
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      int64_t pos = k - j;
      size_t idx = pos >= 0 ? (uint64_t)pos : pos + m;
      uint64_t ws_j, ws_k;
      CUDA_CHECK(cudaMemcpy(&ws_j, &ctx->d_ws_M[j], sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(
          cudaMemcpy(&ws_k, &ctx->d_ws_M[k ? m - k : 0], sizeof(uint64_t),
                     cudaMemcpyDeviceToHost));
      jk_pairs_M[idx] = mont_mul_host(ws_j, ws_k, p, ctx->p_dash);
    }
  }

  uint64_t *jk_sums_M = (uint64_t *)malloc(m * sizeof(uint64_t));
  for (size_t k = 0; k < m; ++k) {
    int64_t pos = k;
    size_t idx = pos >= 0 ? (uint64_t)pos : pos + m;
    uint64_t x = jk_pairs_M[idx];
    int64_t pos2 = -((int64_t)k);
    size_t idx2 = pos2 >= 0 ? (uint64_t)pos2 : pos2 + m;
    uint64_t y = jk_pairs_M[idx2];
    uint64_t sum = x + y;
    if (sum >= p)
      sum -= p;
    jk_sums_M[idx] = sum;
  }

  CUDA_CHECK(cudaMalloc(&ctx->d_jk_sums_M, m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_sums_M, jk_sums_M, m * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  uint64_t *jk_prod_M = (uint64_t *)malloc(m * sizeof(uint64_t));
  for (size_t k = 0; k < m; ++k) {
    int64_t pos = k;
    size_t idx = pos >= 0 ? (uint64_t)pos : pos + m;
    uint64_t sum_inv = mont_inv_host(jk_sums_M[idx], ctx->r3, p, ctx->p_dash);
    jk_prod_M[idx] = mont_mul_host(jk_pairs_M[idx], sum_inv, p, ctx->p_dash);
  }

  CUDA_CHECK(cudaMalloc(&ctx->d_jk_prod_M, m * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(ctx->d_jk_prod_M, jk_prod_M, m * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  free(jk_pairs_M);
  free(jk_sums_M);
  free(jk_prod_M);

  // Natural numbers and inverses
  uint64_t *nat_M = (uint64_t *)malloc((n + 1) * sizeof(uint64_t));
  for (size_t i = 0; i <= n; ++i)
    nat_M[i] = mont_mul_host(i, ctx->r2, p, ctx->p_dash);

  CUDA_CHECK(cudaMalloc(&ctx->d_nat_M, (n + 1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(ctx->d_nat_M, nat_M, (n + 1) * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  uint64_t *nat_inv_M = (uint64_t *)malloc((n + 1) * sizeof(uint64_t));
  nat_inv_M[0] = 0;
  for (size_t k = 1; k <= n; ++k)
    nat_inv_M[k] = mont_inv_host(nat_M[k], ctx->r3, p, ctx->p_dash);

  CUDA_CHECK(cudaMalloc(&ctx->d_nat_inv_M, (n + 1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(ctx->d_nat_inv_M, nat_inv_M, (n + 1) * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  free(nat_M);
  free(nat_inv_M);

  // Factorials and factorial inverses
  uint64_t *fact_M = (uint64_t *)malloc((n + 1) * sizeof(uint64_t));
  fact_M[0] = ctx->r;
  for (size_t i = 1; i < n + 1; ++i) {
    uint64_t nat_i;
    CUDA_CHECK(cudaMemcpy(&nat_i, &ctx->d_nat_M[i], sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    fact_M[i] = mont_mul_host(fact_M[i - 1], nat_i, p, ctx->p_dash);
  }

  CUDA_CHECK(cudaMalloc(&ctx->d_fact_M, (n + 1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(ctx->d_fact_M, fact_M, (n + 1) * sizeof(uint64_t),
                        cudaMemcpyHostToDevice));

  uint64_t *fact_inv_M = (uint64_t *)malloc((n + 1) * sizeof(uint64_t));
  fact_inv_M[n] = mont_inv_host(fact_M[n], ctx->r3, p, ctx->p_dash);
  for (size_t i = n; i; --i) {
    uint64_t nat_i;
    CUDA_CHECK(cudaMemcpy(&nat_i, &ctx->d_nat_M[i], sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));
    fact_inv_M[i - 1] = mont_mul_host(fact_inv_M[i], nat_i, p, ctx->p_dash);
  }

  CUDA_CHECK(cudaMalloc(&ctx->d_fact_inv_M, (n + 1) * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(ctx->d_fact_inv_M, fact_inv_M,
                        (n + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice));

  free(fact_M);
  free(fact_inv_M);

  // Powers of r (for fast_pow_2)
  ctx->n_rs = (n_args * n_args + 63) / 64 + 3;
  uint64_t *rs = (uint64_t *)malloc(sizeof(uint64_t) * ctx->n_rs);
  rs[0] = 1;
  for (size_t i = 1; i < ctx->n_rs; i++)
    rs[i] = mont_mul_host(rs[i - 1], ctx->r2, p, ctx->p_dash);

  CUDA_CHECK(cudaMalloc(&ctx->d_rs, sizeof(uint64_t) * ctx->n_rs));
  CUDA_CHECK(cudaMemcpy(ctx->d_rs, rs, sizeof(uint64_t) * ctx->n_rs,
                        cudaMemcpyHostToDevice));

  free(rs);

  return ctx;
}

void gpu_ctx_free(gpu_ctx_t *ctx) {
  if (!ctx)
    return;
  CUDA_CHECK(cudaFree(ctx->d_ws_M));
  CUDA_CHECK(cudaFree(ctx->d_jk_prod_M));
  CUDA_CHECK(cudaFree(ctx->d_jk_sums_M));
  CUDA_CHECK(cudaFree(ctx->d_nat_M));
  CUDA_CHECK(cudaFree(ctx->d_nat_inv_M));
  CUDA_CHECK(cudaFree(ctx->d_fact_M));
  CUDA_CHECK(cudaFree(ctx->d_fact_inv_M));
  CUDA_CHECK(cudaFree(ctx->d_rs));
  free(ctx);
}

uint64_t gpu_process_batch(gpu_ctx_t *ctx, const size_t *vecs, size_t n_vecs) {
  if (n_vecs == 0)
    return 0;

  size_t *d_vecs;
  uint64_t *d_results;

  size_t vecs_size = n_vecs * ctx->m * sizeof(size_t);
  CUDA_CHECK(cudaMalloc(&d_vecs, vecs_size));
  CUDA_CHECK(cudaMemcpy(d_vecs, vecs, vecs_size, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_results, n_vecs * sizeof(uint64_t)));

  int block_size = 256;
  int num_blocks = (n_vecs + block_size - 1) / block_size;

  compute_kernel<<<num_blocks, block_size>>>(
      d_vecs, n_vecs, d_results, ctx->n, ctx->n_args, ctx->m, ctx->m_half,
      ctx->p, ctx->p_dash, ctx->r, ctx->r2, ctx->r3, ctx->d_ws_M,
      ctx->d_jk_prod_M, ctx->d_jk_sums_M, ctx->d_nat_M, ctx->d_nat_inv_M,
      ctx->d_fact_M, ctx->d_fact_inv_M, ctx->d_rs, ctx->jack_mode);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Reduce results
  while (n_vecs > 1) {
    int reduce_blocks = (n_vecs + block_size - 1) / block_size;
    reduce_kernel<<<reduce_blocks, block_size, block_size * sizeof(uint64_t)>>>(
        d_results, n_vecs, ctx->p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    n_vecs = reduce_blocks;
  }

  uint64_t result;
  CUDA_CHECK(
      cudaMemcpy(&result, d_results, sizeof(uint64_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_vecs));
  CUDA_CHECK(cudaFree(d_results));

  return result;
}
