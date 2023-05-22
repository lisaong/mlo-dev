// Double-buffered banded matrix multiplication
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
// #include <cuda/pipeline>
#include <cuda_runtime.h>

// #define DEBUG 1
#include "constants.h"
#include "utils.h"

// https://developer.nvidia.com/blog/cooperative-groups/
namespace cg = cooperative_groups;

enum class Strategy { SynchronousCopy = 0, AsynchronousCopy = 1 };

__global__ void bandedMatMul_syncCopy(int n0, int n1, int n2, float *t0,
                                      const float *t1, const float *t2,
                                      int tile) {

  int i, j, k, jj;

  auto cta = cg::this_thread_block();

  // load the t0 and t1 sub-matrices into shared memory
  extern __shared__ float t0_s[];
  float *t1_s = &t0_s[cta.size() * tile];

  const auto rowStart = blockIdx.x * blockDim.x + threadIdx.x;
  const auto rowStride = blockDim.x * gridDim.x;
  const auto colStart = blockIdx.y * blockDim.y + threadIdx.y;
  const auto colStride = blockDim.y * gridDim.y;

  for (i = rowStart; i < n0; i += rowStride) {
    for (j = colStart; j * tile < n1; j += colStride) {

      // for each thread, copy a column-tile of t0 and t1 into shared memory
      for (jj = 0; jj < tile; ++jj) {
        const auto smemIdx =
            threadIdx.x * blockDim.y * tile + threadIdx.y * tile + jj;
        t0_s[smemIdx] = t0[i * n1 + j * tile + jj];
        t1_s[smemIdx] = t1[i * n2 + j * tile + jj];
      }
    }
  }
  cta.sync();

  // compute
  for (i = rowStart; i < n0; i += rowStride) {
    for (j = colStart; j * tile < n1; j += colStride) {
      for (jj = 0; jj < tile; ++jj) {
        const auto smemIdx =
            threadIdx.x * blockDim.y * tile + threadIdx.y * tile + jj;

        // treat t2 as column major
        for (k = 0; i + k < n1; ++k) {
          t0_s[smemIdx] += t1_s[smemIdx] * t2[(i + k) + (j * tile + jj) * n2];
        }
        cta.sync();

        // write back to global memory
        t0[i * n1 + j * tile + jj] = t0_s[smemIdx];
      }
    }
  }
}

__global__ void bandedMatMul_asyncCopy(int n0, int n1, int n2, float *t0,
                                       const float *t1, const float *t2,
                                       int tile) {

  int i, j, k, jj;

  auto cta = cg::this_thread_block();

  // load the t0 and t1 sub-matrices into shared memory
  extern __shared__ float t0_s[];
  float *t1_s = &t0_s[cta.size() * tile];

  const auto rowStart = blockIdx.x * blockDim.x + threadIdx.x;
  const auto rowStride = blockDim.x * gridDim.x;
  const auto colStart = blockIdx.y * blockDim.y + threadIdx.y;
  const auto colStride = blockDim.y * gridDim.y;

  i = rowStart;
  j = colStart;
  cg::memcpy_async(cta, t0_s, &t0[i * n1 + j * tile],
                   sizeof(float) * tile * cta.size());
  cg::memcpy_async(cta, t1_s, &t1[i * n2 + j * tile],
                   sizeof(float) * tile * cta.size());
  cg::wait(cta);

  for (i = rowStart; i < n0; i += rowStride) {
    for (j = colStart; j * tile < n1; j += colStride) {

      // compute
      for (jj = 0; jj < tile; ++jj) {
        const auto smemIdx =
            threadIdx.x * blockDim.y * tile + threadIdx.y * tile + jj;

        // treat t2 as column major
        for (k = 0; i + k < n1; ++k) {
          t0_s[smemIdx] += t1_s[smemIdx] * t2[(i + k) + (j * tile + jj) * n2];
        }

        // write back to global memory
        t0[i * n1 + j * tile + jj] = t0_s[smemIdx];
      }
    }
  }
}

void run(int deviceId, Strategy strategy) {

  const int n0 = N; // n0: number of rows in T0 and T1
  const int n1 = N; // n1: number of columns in T0 and T2
  const int n2 = N; // n2: inner or shared dimension, i.e.
                    //     number of columns in T1 and number of rows in T2

  Matrix<float> T0(n0, n1);                                 // output
  BandedMatrix<float> T1(n0, kBandDim);                     // input
  Matrix<float> T2(T1.columns(), n1, /*columnMajor*/ true); // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

  // Initialize
  dim3 threads(kBlockDimX, kMaxBlockDim / kBlockDimX, 1);
  dim3 blocks(ceildiv(n0, threads.x), ceildiv(n1, threads.y), 1);
  fillMatrices(T0, T1, T2, blocks, threads, deviceId);

  // Verify
  // shared memory: [t0 sub-matrix, t1 sub-matrix]
  threads.y = ceildiv(n1, threads.y * kTile);
  uint32_t smemSize = threads.x * threads.y * sizeof(float) * 2 * kTile;

  switch (strategy) {
  case Strategy::SynchronousCopy:
    bandedMatMul_syncCopy<<<blocks, threads, smemSize>>>(
        n0, n1, n2, T0.data, T1.data, T2.data, kTile);
    break;
  case Strategy::AsynchronousCopy:
    bandedMatMul_asyncCopy<<<blocks, threads, smemSize>>>(
        n0, n1, n2, T0.data, T1.data, T2.data, kTile);
    break;
  default:
    throw std::runtime_error("Unknown strategy");
  };

  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  if (checkCorrectness(n0, n1, n2, T0, T1, T2)) {

    // Benchmark
    cudaEvent_t _start;
    cudaEvent_t _stop;
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);

    // Try different block sizes
    std::cout << "GridDim,BlockDim,FLOPS,GFLOPS" << std::endl;

    for (uint32_t blockDim = kBlockDimX; blockDim <= kBlockDimXMax;
         blockDim += kBlockDimXStep) {

      threads.x = blockDim;
      threads.y = ceildiv(kMaxBlockDim, blockDim * kTile);
      blocks.x = ceildiv(n0, threads.x);
      blocks.y = ceildiv(n1, threads.y);
      smemSize = threads.x * threads.y * sizeof(float) * 2 * kTile;

      try {
        double elapsedTimeMilliseconds = 0.0f;
        uint64_t iterations = 0;
        float duration = 0.0f;

        // Runs the function until 10 seconds has elapsed
        cudaEventRecord(_start);
        while (elapsedTimeMilliseconds < kTimelimit) {

          switch (strategy) {
          case Strategy::SynchronousCopy:
            bandedMatMul_syncCopy<<<blocks, threads, smemSize>>>(
                n0, n1, n2, T0.data, T1.data, T2.data, kTile);
            break;
          case Strategy::AsynchronousCopy:
            bandedMatMul_asyncCopy<<<blocks, threads, smemSize>>>(
                n0, n1, n2, T0.data, T1.data, T2.data, kTile);
            break;
          default:
            break;
          };

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
        std::cout << blocks.x << "," << threads.x << "," << flops << ","
                  << flops / 1e9 << std::endl;
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
  Strategy strategy = Strategy::AsynchronousCopy;

  if (argc > 1) {
    deviceId = atoi(argv[1]);
    CHECK(cudaSetDevice(deviceId));
  } else {
    CHECK(cudaGetDevice(&deviceId));
  }
  std::cout << "Using device " << deviceId << std::endl;

  if (argc > 2) {
    strategy = static_cast<Strategy>(atoi(argv[2]));
  }
  std::cout << "Using strategy " << static_cast<int>(strategy) << std::endl;

  run(deviceId, strategy);
  return 0;
}