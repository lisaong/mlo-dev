// Naive banded matrix multiplication
#include <cstdint>
#include <cuda_runtime.h>

// #define DEBUG 1
#include "constants.h"
#include "utils.h"

__global__ void bandedMatMul_Naive(int n0, int n1, int n2, float *t0,
                                   const float *t1, const float *t2) {

  int i, j, k;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < n0;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < n1;
         j += blockDim.y * gridDim.y) {
      for (k = 0; k < n2 && (i + k) < n0; ++k) {
        t0[i * n1 + j] += t1[i * n2 + k] * t2[(i + k) * n1 + j];
      }
    }
  }
}

void run(int deviceId) {

  const int n0 = N; // n0: number of rows in T0 and T1
  const int n1 = N; // n1: number of columns in T0 and T2
  const int n2 = N; // n2: inner or shared dimension, i.e.
                    //     number of columns in T1 and number of rows in T2

  Matrix T0(n0, n1);             // output
  BandedMatrix T1(n0, kBandDim); // input
  Matrix T2(T1.columns(), n1);   // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

  // Initialize
  dim3 threadsInit(kBlockDim, kBlockDim, 1);
  dim3 blocksInit(n0 / threadsInit.x, n1 / threadsInit.y, 1);
  initWith<<<blocksInit, threadsInit>>>(11.0f, T0.data, T0.rows(),
                                        T0.columns());
  initBandedWith<<<blocksInit, threadsInit>>>(22.0f, T1.data, T1.rows(),
                                              T1.columns(), T1.band());
  initWith<<<blocksInit, threadsInit>>>(33.0f, T2.data, T2.rows(),
                                        T2.columns());
  CHECK(cudaDeviceSynchronize());

  // Verify
  bandedMatMul_Naive<<<blocksInit, threadsInit>>>(n0, n1, n2, T0.data, T1.data,
                                                  T2.data);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  if (checkCorrectness(n0, n1, n2, T0, T1, T2)) {

    // Benchmark
    cudaEvent_t _start;
    cudaEvent_t _stop;
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);

    // Try different block sizes
    for (uint32_t blockDim = kBlockDim; blockDim <= kMaxBlockDim;
         blockDim += kBlockDimStep) {

      dim3 threads(blockDim, blockDim, 1);
      dim3 blocks(n0 / threads.x, n1 / threads.y, 1);

      try {
        double elapsedTimeMilliseconds = 0.0f;
        uint64_t iterations = 0;
        float duration = 0.0f;

        // Runs the function until 10 seconds has elapsed
        cudaEventRecord(_start);
        while (elapsedTimeMilliseconds < kTimelimit) {
          bandedMatMul_Naive<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
                                                  T2.data);

          CHECK(cudaGetLastError());
          CHECK(cudaDeviceSynchronize());

          cudaEventRecord(_stop);
          cudaEventSynchronize(_stop);
          cudaEventElapsedTime(&duration, _start, _stop);
          elapsedTimeMilliseconds += duration;
          iterations++;
        }

        const double flops = iterations * kNumberOfOps /
                             (elapsedTimeMilliseconds / kMillisecondsInSeconds);
        std::cout << "Blocksize: " << blockDim << ", Iterations: " << iterations
                  << ", FLOPS: " << flops << ", GFLOPS: " << flops / 1e9
                  << std::endl;
      } catch (const std::exception &e) {
        std::cout << "Skipping Blocksize: " << blockDim << ", " << e.what()
                  << std::endl;
        continue;
      }
    }

    cudaEventDestroy(_start);
    cudaEventDestroy(_stop);
  }

  cudaFree(T0.data);
  cudaFree(T1.data);
  cudaFree(T2.data);
}

int main(int argc, const char **argv) {
  int deviceId;

  if (argc > 1) {
    deviceId = atoi(argv[1]);
    CHECK(cudaSetDevice(deviceId));
  } else {
    CHECK(cudaGetDevice(&deviceId));
  }
  std::cout << "Using device " << deviceId << std::endl;

  run(deviceId);
  return 0;
}