// Double-buffered banded matrix multiplication
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
// #include <cuda/pipeline>
#include <cuda_runtime.h>

// #define T2_SMEM 1
#define DEBUG 1
#include "constants.h"
#include "utils.h"

// https://developer.nvidia.com/blog/cooperative-groups/
namespace cg = cooperative_groups;

enum class Strategy { SynchronousCopy = 0, AsynchronousCopy = 1 };

__global__ void bandedMatMul_syncCopy(int n0, int n1, int n2, float *t0,
                                      const float *t1, const float *t2,
                                      int tileK) {

  auto cta = cg::this_thread_block();

  extern __shared__ float t0_s[];                   // blockDim.x * blockDim.y
  float *t1_s = &t0_s[cta.size()];                  // blockDim.x * tileK
  float *t2_s = &t1_s[cta.dim_threads().x * tileK]; // tileK * blockDim.y

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n0;
       i += blockDim.x * gridDim.x) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n1;
         j += blockDim.y * gridDim.y) {

      // T0: prepare the result tile
      // Each block copies blockDim.x x blockDim.y entries, one entry per thread
      t0_s[threadIdx.x * blockDim.y + threadIdx.y] = t0[i * n1 + j];

      // Due to shared memory limitations, we cannot fit complete rows or
      // columns of T1 and T2 per block:
      //   T1: only (blockDim.x, tileK) shared memory available
      //   T2: only (tileK, blockDim.y) shared memory available
      // Perform the copying and multiplication per tile, then accumulate
      // the results in a blockDim.x by blockDim.y T0 tile
      const auto numTilesPerBlock = n1 / tileK;
      const auto colsPerThread = tileK / blockDim.y;

      for (int t = 0; t < numTilesPerBlock; ++t) {
        const auto tileOffset = t * tileK;

        // T1: Each block fills the (blockDim.x, tileK) shared memory
        // Each thread fills a (1, colsPerThread) row
        for (int k = 0; k < colsPerThread; ++k) {
          const auto t1ThreadY = threadIdx.y * colsPerThread + k;
          const auto col = tileOffset + t1ThreadY;
          const auto idx = i * n1 + col;
          const auto sIdx = threadIdx.x * tileK + t1ThreadY;
          t1_s[sIdx] = t1[idx];
        }

#if T2_SMEM
        // T2: Each block fills the (tileK, blockDim.y) shared memory
        // Each thread fills a (rowsPerThread, 1) column, offset by i
        const auto rowsPerThread = tileK / blockDim.x;
        const auto t2ThreadY = threadIdx.y;

        for (int k = 0; k < rowsPerThread; ++k) {
          const auto t2ThreadX = threadIdx.x * rowsPerThread + k;
          const auto row = i + tileOffset + t2ThreadX;
          const auto col = j;
          if (row < n0) {
            // column major layout
            const auto idx = col * n1 + row;
            const auto sIdx = t2ThreadY * tileK + t2ThreadX;
            t2_s[sIdx] = t2[idx];
          }
        }
#endif // T2_SMEM
        cta.sync();

        // Each block multiplies (blockDim.x, tileK) with (tileK, blockDim.y)
        // and accumulates the results into T0
        // Each thread multiplies (1, tileK) with (tileK, 1) for a particular
        // (i, j) entry in T0
        const auto sIdx = threadIdx.x * blockDim.y + threadIdx.y;
        for (int k = 0; k < tileK; ++k) {
#if T2_SMEM
          t0_s[sIdx] +=
              t1_s[threadIdx.x * tileK + k] * t2_s[threadIdx.y * tileK + k];
#else
          // reverse map T2's local row coordinate to global row coordinate
          // local: k, global: t * tileK + k
          const auto row = tileOffset + k;
          if (i + row < n0) {
            t0_s[sIdx] +=
                t1_s[threadIdx.x * tileK + k] * t2[(i + row) + j * n2];
          }
#endif // T2_SMEM
        }
      }

      t0[i * n1 + j] = t0_s[threadIdx.x * blockDim.y + threadIdx.y];
    }
  }
}

__global__ void bandedMatMul_asyncCopy(int n0, int n1, int n2, float *t0,
                                       const float *t1, const float *t2,
                                       int tileK) {

  auto cta = cg::this_thread_block();

  extern __shared__ float t0_s[];                   // blockDim.x * blockDim.y
  float *t1_s = &t0_s[cta.size()];                  // blockDim.x * tileK
  float *t2_s = &t1_s[cta.dim_threads().x * tileK]; // tileK * blockDim.y

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n0;
       i += blockDim.x * gridDim.x) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < n1;
         j += blockDim.y * gridDim.y) {

      // T0: prepare the result tile
      // Each block copies blockDim.x x blockDim.y entries, one entry per thread
      t0_s[threadIdx.x * blockDim.y + threadIdx.y] = t0[i * n1 + j];

      // Due to shared memory limitations, we cannot fit complete rows or
      // columns of T1 and T2 per block:
      //   T1: only (blockDim.x, tileK) shared memory available
      //   T2: only (tileK, blockDim.y) shared memory available
      // Perform the copying and multiplication per tile, then accumulate
      // the results in a blockDim.x by blockDim.y T0 tile
      const auto numTilesPerBlock = n1 / tileK;
      const auto colsPerThread = tileK / blockDim.y;

      for (int t = 0; t < numTilesPerBlock; ++t) {
        const auto tileOffset = t * tileK;

        // T1: Each block fills the (blockDim.x, tileK) shared memory
        // Each thread fills a (1, colsPerThread) row
        const auto t1ThreadY = threadIdx.y * colsPerThread;
        auto col = tileOffset + t1ThreadY;
        auto idx = i * n1 + col;
        auto sIdx = threadIdx.x * tileK + t1ThreadY;

        cg::memcpy_async(cta, t1_s + sIdx, t1 + idx,
                         sizeof(float) * colsPerThread);
        cg::wait(cta);

#if T2_SMEM
        {
          // T2: Each block fills the (tileK, blockDim.y) shared memory
          // Each thread fills a (rowsPerThread, 1) column, offset by i
          const auto rowsPerThread = tileK / blockDim.x;
          const auto t2ThreadY = threadIdx.y;
          const auto t2ThreadX = threadIdx.x * rowsPerThread;
          row = i + tileOffset + t2ThreadX;
          col = j;
          if ((row + rowsPerThread) < n0) {
            // column major layout
            idx = col * n1 + row;
            sIdx = t2ThreadY * tileK + t2ThreadX;
            cg::memcpy_async(cta, t2_s + sIdx, t2 + idx,
                             sizeof(float) * rowsPerThread);
          }
          cg::wait(cta);
        }

#endif // T2_SMEM

        // Each block multiplies blockDim.x by tileK with tileK by blockDim.y
        // and accumulates the results into T0 Each thread multiplies 1 x tileK
        // with tileX by 1 and accumulates the results into T0
        sIdx = threadIdx.x * blockDim.y + threadIdx.y;
        for (int k = 0; k < tileK; ++k) {
#if T2_SMEM
          t0_s[sIdx] +=
              t1_s[threadIdx.x * tileK + k] * t2_s[threadIdx.y * tileK + k];
#else
          // reverse map T2's local row coordinate to global row coordinate
          // local: k, global: t * tileK + k
          const auto row = tileOffset + k;
          if (i + row < n0) {
            t0_s[sIdx] +=
                t1_s[threadIdx.x * tileK + k] * t2[(i + row) + j * n2];
          }
#endif // T2_SMEM
        }
        cta.sync();
      }

      t0[i * n1 + j] = t0_s[threadIdx.x * blockDim.y + threadIdx.y];
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
  // dim3 threads(kBlockDimX, kMaxBlockDim / kBlockDimX, 1);
  dim3 threads(16, 2, 1);
  dim3 blocks(n0 / threads.x, n1 / threads.y, 1);
  fillMatrices(T0, T1, T2, blocks, threads, deviceId);

  // divide the inner dimension (k) among threads.y
  int tileK = n1 / threads.y;

  // hold tiles of T0, T1, and T2 in shared memory
  uint32_t smemSize = threads.x * threads.y * sizeof(float) +
                      threads.x * tileK * sizeof(float) +
                      tileK * threads.y * sizeof(float);

  // Verify
  switch (strategy) {
  case Strategy::SynchronousCopy:
    bandedMatMul_syncCopy<<<blocks, threads, smemSize>>>(
        n0, n1, n2, T0.data, T1.data, T2.data, tileK);
    break;
  case Strategy::AsynchronousCopy:
    bandedMatMul_asyncCopy<<<blocks, threads, smemSize>>>(
        n0, n1, n2, T0.data, T1.data, T2.data, tileK);
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
      threads.y = kMaxBlockDim / blockDim;
      blocks.x = n0 / threads.x;
      blocks.y = n1 / threads.y;
      tileK = n1 / threads.y;
      smemSize = threads.x * threads.y * sizeof(float) +
                 threads.x * tileK * sizeof(float) +
                 tileK * threads.y * sizeof(float);

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
                n0, n1, n2, T0.data, T1.data, T2.data, tileK);
            break;
          case Strategy::AsynchronousCopy:
            bandedMatMul_asyncCopy<<<blocks, threads, smemSize>>>(
                n0, n1, n2, T0.data, T1.data, T2.data, tileK);
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