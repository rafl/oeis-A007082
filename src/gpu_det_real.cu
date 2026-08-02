
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
  u_int32_t * out_scaling_factors,
     fld_t p,
                                           fld_t p_dash, fld_t r,
                                           fld_t r3,
                                           int num_matricies
                                        
                                        ) {

int workIdx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    // Compute determinant via Gaussian elimination
    fld_t det = r, scaling_factor = r;

    // On this branch pragma unroll seems to make little difference
    // #pragma unroll
    for (size_t k = 0; k < DIM; ++k) {
      // Find pivot
      fld_t pivot = data[(k * DIM + k) * blockDim.x + threadIdx.x];
      det = d_mont_mul(det, data[(k * DIM + k) * blockDim.x + threadIdx.x], p, p_dash);

      // Elimination
      // #pragma unroll
      for (size_t i = k + 1; i < DIM; ++i) {
        scaling_factor = d_mont_mul(scaling_factor, pivot, p, p_dash);
        fld_t multiplier = data[(i * DIM + k) * blockDim.x + threadIdx.x];
        for (size_t j = k; j < DIM; ++j) {
          data[(i * DIM + j) * blockDim.x + threadIdx.x] = d_mont_mul_sub(data[(i * DIM + j) * blockDim.x + threadIdx.x], pivot, data[(k * DIM + j) * blockDim.x + threadIdx.x],
                                          multiplier, p, p_dash);
        }
      }
    }

    out[workIdx] = det; // d_mont_mul(det, d_mont_inv(scaling_factor, r3, p, p_dash), p, p_dash);
    out_scaling_factors[workIdx] = scaling_factor;
  }


// best recorded time:
// Elapsed time: 2.394943119 seconds
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
    u_int32_t *device_scaling_factor;
    CUDA_CHECK(cudaMalloc(&device_result, n_matricies * sizeof(u_int32_t)));
    CUDA_CHECK(cudaMalloc(&device_scaling_factor, n_matricies * sizeof(u_int32_t)));
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < 1000; i++)
    {
    // 3. Launch the kernel using the triple angle bracket execution syntax
    det_mod_p_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_buffer, device_result, device_scaling_factor, p, p_dash, r, r3, n_matricies);
    }

    //just for timing:
    CUDA_CHECK(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Elapsed time: %.9f seconds\n", elapsed);

    CUDA_CHECK(cudaMemcpy(results, device_result, n_matricies * sizeof(u_int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results, device_scaling_factor, n_matricies * sizeof(u_int32_t), cudaMemcpyDeviceToHost));

    // Copy the result back from device to host
    // cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
}