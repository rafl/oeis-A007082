
#include <cuda_runtime.h>

// #include "debug.h"
// #include "gpu_det.h"

#include "gpu_det_real.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "maths.h"
#include "mss.h"
#include "primes.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      abort();                                                                 \
    }                                                                          \
  } while (0)

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


#define SIZE 20
// #define fld_t u_int32_t
#define DIM SIZE

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


__global__ void det_mod_p_kernel(u_int32_t *data, u_int32_t* out,
     fld_t p,
                                           fld_t p_dash, fld_t r,
                                           fld_t r3,
                                           int num_matricies
                                        
                                        ) {

int workIdx = ((blockIdx.x * blockDim.x) + threadIdx.x);
    fld_t * A = data + (SIZE * SIZE) * workIdx;
    if (workIdx >= num_matricies)
    {
        return;
    }
   
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

    out[workIdx] = d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p, p_dash);

    
  }



void det_mod_p_gpu(u_int32_t const * values, uint32_t * results, u_int32_t n_matricies,      fld_t p,
                                           fld_t p_dash, fld_t r,
                                           fld_t r3)
{
    // int n_sms= 84;
    // int threads_per_sm = 1536;
    int threadsPerBlock = 256;
    // uint32_t blocksPerGrid = (threads_per_sm / threadsPerBlock) * n_sms;
    int blocksPerGrid = (n_matricies + threadsPerBlock - 1)/ threadsPerBlock;
    u_int32_t *device_buffer;
    CUDA_CHECK(cudaMalloc(&device_buffer, SIZE * SIZE * n_matricies * sizeof(u_int32_t)));
    CUDA_CHECK(cudaMemcpy(device_buffer, values, SIZE * SIZE * n_matricies * sizeof(u_int32_t), cudaMemcpyHostToDevice));

    u_int32_t *device_result;
    CUDA_CHECK(cudaMalloc(&device_result, n_matricies * sizeof(u_int32_t)));


    // 3. Launch the kernel using the triple angle bracket execution syntax
    det_mod_p_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_buffer, device_result, p, p_dash, r, r3, n_matricies);

    CUDA_CHECK(cudaMemcpy(results, device_result, n_matricies * sizeof(u_int32_t), cudaMemcpyDeviceToHost));

    // Copy the result back from device to host
    // cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
}