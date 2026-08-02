
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

// __device__ inline fld_t d_extended_euclidean(fld_t a, fld_t b) {
//   fld_t r0 = a;
//   fld_t r1 = b;
//   fld_t s0 = 1;
//   fld_t s1 = 0;
//   fld_t spare;
//   size_t n = 0;
//   while (r1) {
//     fld_t q = r0 / r1;
//     spare = r0 % r1;
//     r0 = r1;
//     r1 = spare;
//     spare = s0 + q * s1;
//     s0 = s1;
//     s1 = spare;
//     ++n;
//   }
//   if (n % 2)
//     s0 = b - s0;
//   return s0;
// }

// __device__ inline fld_t d_mont_inv(fld_t x, fld_t r3, fld_t p, fld_t p_dash) {
//   fld_t inv = d_extended_euclidean(x, p);
//   return d_mont_mul(r3, inv, p, p_dash);
// }


__global__ void det_mod_p_kernel(u_int32_t *data, u_int32_t* out, uint32_t* out_sf,
  // TODO return scaling factors as well
     fld_t p,
                                           fld_t p_dash, fld_t r,
                                           fld_t r3,
                                           int num_matricies
                                        
                                        ) {
    // __shared__ int scalingFactors[SIZE * SIZE];
    // __shared__ int dets[SIZE * SIZE];

        // Compute determinant via Gaussian elimination
    fld_t det = r, scaling_factor = r;
    fld_t i = threadIdx.x / DIM;
    fld_t j = threadIdx.x % DIM;

for (int lp = 0; lp <= 256; lp++)
{
    int workIdx = ((blockIdx.x * blockDim.x) + lp);
    fld_t * A = data + (SIZE * SIZE) * workIdx;
    if (workIdx >= num_matricies)
    {
        // This is not legit
        return;
    }

    // Let's take this at it's absolute best and try to beat it
    if (threadIdx.x < DIM * DIM)
    {
      for (size_t k = 0; k < DIM; ++k) {
        fld_t pivot = A[k * DIM + k];
        // Every thread computing det is silly... whatever
        det = d_mont_mul(det, A[k * DIM + k], p, p_dash);

        // Elimination
        if (i >= k +1) {
        // for (size_t i = k + 1; i < DIM; ++i) {
          // Every thread computing scaling factor is silly...
          // This scaling factor needs to be raise to some power...
          scaling_factor = d_mont_mul(scaling_factor, pivot, p, p_dash);
          fld_t multiplier = A[i * DIM + k];
          // for (size_t j = k; j < DIM; ++j) {
          if (j >= k) {
            A[i * DIM + j] = d_mont_mul_sub(A[i * DIM + j], pivot, A[k * DIM + j],
                                            multiplier, p, p_dash);
          }
      }

      __syncthreads();
    }
    if (threadIdx.x == 0) {
      out[workIdx] = det;
      out_sf[workIdx] = scaling_factor;
    }
  }
  }
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