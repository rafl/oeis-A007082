
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

#define WIDE_LOAD_SIZE 4

struct alignas(WIDE_LOAD_SIZE * sizeof(u_int32_t)) WideLoad {
  u_int32_t data[WIDE_LOAD_SIZE];
};

union alignas((SIZE / WIDE_LOAD_SIZE) * sizeof(uint32_t)) Row {
  WideLoad loader[SIZE / WIDE_LOAD_SIZE];
  uint32_t data[SIZE];
};

#define MATRIX_EL(row_idx, col_idx) (row_idx * DIM + col_idx)

__global__ void det_mod_p_kernel(u_int32_t *data, u_int32_t* out, uint32_t* out_sf,
  // TODO return scaling factors as well
     fld_t p,
                                           fld_t p_dash, fld_t r,
                                           fld_t r3,
                                           int num_matricies
                                        
                                        )                                         
{
    // mask not right for other sizes
    // fld_t mask = (threadIdx.x % 32 < SIZE) ?  0xffff : 0xffff << SIZE;
    fld_t row_idx = threadIdx.x % DIM;
        Row myRow;

  for (int lp = 0; lp < SIZE; lp++)
  {
    // Again just assume memory layout works out for this
    // Two matricies per loop
    // We've got 256 threads... we're suppoed to process 256 elements
    // That's 8 warps, so we're doing 16 matricies at a time
    // so step in units of 16 matricies per loop
    // and 256 matricies per block index

    int workIdx = ((blockIdx.x * blockDim.x) + lp * SIZE);

    // Whatever - just assume the data is in the magic layout to make this legit:
    #pragma unroll
    for (int i = 0; i < (SIZE / WIDE_LOAD_SIZE); i++)
    {
      myRow.loader[i] = *reinterpret_cast<WideLoad*>(data + workIdx + ((threadIdx.x + i * blockDim.x) * WIDE_LOAD_SIZE));
    }

    // #pragma unroll  
    // for (int i = 0; i < (SIZE); i++)
    // {
    //   myRow.data[i] = data[workIdx + ((threadIdx.x + i * blockDim.x))];
    // }

    // Compute determinant via Gaussian elimination
    fld_t det = r, scaling_factor = r;
   
    // fld_t * A = data + (SIZE * SIZE) * workIdx;

    #pragma unroll  
    for (size_t k = 0; k < DIM; ++k) {
      fld_t pivot = __shfl_sync(0xffffffff, myRow.data[k], k, SIZE);
      det = d_mont_mul(det, pivot, p, p_dash);

      // Elimination
      // IDK if this condition helps?
      // if (row_idx >= k +1) {
      // for (size_t i = k + 1; i < DIM; ++i) {
        // Every row computing scaling factor is silly...
        // This scaling factor needs to be raise to some power...
        scaling_factor = d_mont_mul(scaling_factor, pivot, p, p_dash);
        fld_t multiplier = myRow.data[k];
        #pragma unroll  
        for (size_t col_idx = k; col_idx < DIM; ++col_idx) {
            fld_t pivotRowVal = __shfl_sync(0xffffffff, myRow.data[col_idx], k, SIZE);
   
            if (row_idx > k) {
            myRow.data[col_idx] = d_mont_mul_sub(myRow.data[col_idx], pivot, pivotRowVal,
                                            multiplier, p, p_dash);
            }
          }
    }

    // TODO the reduction better
    if (row_idx == 0) {
      out[workIdx] = det;
      out_sf[workIdx] = scaling_factor;
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