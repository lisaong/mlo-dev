// Double-buffered banded matrix multiplication
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cuda/barrier>
#include <cuda_runtime.h>

// #define DEBUG 1
#include "constants.h"
#include "utils.h"

// https://developer.nvidia.com/blog/cooperative-groups/
// https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/
namespace cg = cooperative_groups;

enum class Strategy {
  SynchronousCopy = 0,
  AsynchronousCopy = 1,
  AsynchronousCopyBarriers = 2
};

// TODOs:
//  - T2 block copies into shared memory, possible?

__global__ void bandedMatMul_syncCopy(int n0, int n1, int n2, float *t0,
                                      const float *t1, const float *t2,
                                      int tileK) {

  auto cta = cg::this_thread_block();

  extern __shared__ float t1_s[];                   // blockDim.x * tileK
  float *t2_s = &t1_s[cta.dim_threads().x * tileK]; // tileK * blockDim.y

  const auto iBegin = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iStep =
      blockDim.x * gridDim.x; // number of threads per grid, if grid_size < n0

  const auto jBegin = blockIdx.y * blockDim.y + threadIdx.y;
  const auto jStep =
      blockDim.y * gridDim.y; // number of threads per grid, if grid_size < n1

  for (int i = iBegin, j = jBegin; i < n0 && j < n1; i += iStep, j += jStep) {
    int k;

    float t0_thread = 0.0f; // holds the result for this thread

    // Due to shared memory limitations, we cannot fit complete rows or
    // columns of T1 per block:
    //   T1: only (blockDim.x, tileK) shared memory available
    // Perform the copying and multiplication per tile, then accumulate
    // the results in a blockDim.x by blockDim.y T0 tile
    const auto numKTilesPerBlock = n2 / tileK;
    const auto colsPerThread = tileK / blockDim.y;

    for (int t = 0; t < numKTilesPerBlock; ++t) {
      const auto kOffset = t * tileK;

      // T1: Each block fills the (blockDim.x, tileK) shared memory
      // Each thread fills a (1, colsPerThread) row
      for (int kk = 0; kk < colsPerThread; ++kk) {
        const auto t1ThreadY = threadIdx.y * colsPerThread + kk;
        k = kOffset + t1ThreadY;
        const auto idx = i * n2 + k;
        const auto sIdx = threadIdx.x * tileK + t1ThreadY;
        t1_s[sIdx] = t1[idx];
      }

      cta.sync();

      // Each block multiplies (blockDim.x, tileK) with (tileK, blockDim.y)
      // and accumulates the results into T0
      // Each thread multiplies (1, tileK) with (tileK, 1) for a particular
      // (i, j) entry in T0
      for (int kk = 0; kk < tileK; ++kk) {
        k = kOffset + kk;
        if (i + k < n0) {
          // reverse map T2's local row coordinate to global row coordinate
          // local: kk, global: t * tileK + kk
          t0_thread += t1_s[threadIdx.x * tileK + kk] * t2[(i + k) + j * n2];
        }
      }
    }

    t0[i * n1 + j] += t0_thread;
  }
}

__global__ void bandedMatMul_asyncCopy(int n0, int n1, int n2, float *t0,
                                       const float *t1, const float *t2,
                                       int tileK) {

  auto cta = cg::this_thread_block();

  extern __shared__ float t1_s[];                   // blockDim.x * tileK
  float *t2_s = &t1_s[cta.dim_threads().x * tileK]; // tileK * blockDim.y

  float t0_thread = 0.0f; // holds the result for this thread

  const auto iBegin = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iStep =
      blockDim.x * gridDim.x; // number of threads per grid, if grid_size < n0

  const auto jBegin = blockIdx.y * blockDim.y + threadIdx.y;
  const auto jStep =
      blockDim.y * gridDim.y; // number of threads per grid, if grid_size < n1

  for (int i = iBegin, j = jBegin; i < n0 && j < n1; i += iStep, j += jStep) {

    // Due to shared memory limitations, we cannot fit complete rows or
    // columns of T1 per block:
    //   T1: only (blockDim.x, tileK) shared memory available
    // Perform the copying and multiplication per tile, then accumulate
    // the results into T0's thread local variable
    const auto numKTilesPerBlock = n2 / tileK;

    for (int t = 0; t < numKTilesPerBlock; ++t) {
      const auto kOffset = t * tileK;

      // T1: Each block fills the (blockDim.x, tileK) shared memory
      // To perform the block-level copying, we'll need to loop over the rows
      // of t1 (1, tileK) because we can only specify contiguous memory
      // to memcpy_async
      for (int ii = 0; ii < blockDim.x; ++ii) {
        // collaboratively copy a row
        const auto sIdxRow = ii * tileK;
        const auto idxRow = (blockIdx.x * blockDim.x + ii) * n2 + kOffset;

        cg::memcpy_async(cta, t1_s + sIdxRow, t1 + idxRow,
                         sizeof(float) * tileK);
      }

      cg::wait(cta);

      // Each block multiplies blockDim.x by tileK with tileK by blockDim.y
      // and accumulates the results into T0 Each thread multiplies 1 x tileK
      // with tileX by 1 and accumulates the results into T0
      for (int k = 0; k < tileK; ++k) {
        // reverse map T2's local row coordinate to global row coordinate
        // local: k, global: t * tileK + k
        const auto row = kOffset + k;
        if (i + row < n0) {
          t0_thread += t1_s[threadIdx.x * tileK + k] * t2[(i + row) + j * n2];
        }
      }
    }

    t0[i * n1 + j] += t0_thread;
  }
}

__global__ void bandedMatMul_asyncCopyBarriers(int n0, int n1, int n2,
                                               float *t0, const float *t1,
                                               const float *t2, int tileK) {

  auto cta = cg::this_thread_block();

  // Create a synchronization object (C++20 barrier)
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;

  if (cta.thread_rank() == 0) {
    init(&barrier, cta.size());
  }
  cta.sync();

  extern __shared__ float t1_s[];                   // blockDim.x * tileK
  float *t2_s = &t1_s[cta.dim_threads().x * tileK]; // tileK * blockDim.y

  float t0_thread = 0.0f; // holds the result for this thread

  const auto iBegin = blockIdx.x * blockDim.x + threadIdx.x;
  const auto iStep =
      blockDim.x * gridDim.x; // number of threads per grid, if grid_size < n0

  const auto jBegin = blockIdx.y * blockDim.y + threadIdx.y;
  const auto jStep =
      blockDim.y * gridDim.y; // number of threads per grid, if grid_size < n1

  for (int i = iBegin, j = jBegin; i < n0 && j < n1; i += iStep, j += jStep) {

    // Due to shared memory limitations, we cannot fit complete rows or
    // columns of T1 per block:
    //   T1: only (blockDim.x, tileK) shared memory available
    // Perform the copying and multiplication per tile, then accumulate
    // the results into T0's thread local variable
    const auto numKTilesPerBlock = n2 / tileK;

    for (int t = 0; t < numKTilesPerBlock; ++t) {
      const auto kOffset = t * tileK;

      // T1: Each block fills the (blockDim.x, tileK) shared memory
      // To perform the block-level copying, we'll need to loop over the rows
      // of t1 (1, tileK) because we can only specify contiguous memory
      // to memcpy_async
      for (int ii = 0; ii < blockDim.x; ++ii) {
        // collaboratively copy a row
        const auto sIdxRow = ii * tileK;
        const auto idxRow = (blockIdx.x * blockDim.x + ii) * n2 + kOffset;

        cuda::memcpy_async(cta, t1_s + sIdxRow, t1 + idxRow,
                           sizeof(float) * tileK, barrier);

        // BUGBUG: hangs for certain block sizes
        barrier.arrive_and_wait(); // Wait for all copies to complete
      }

      // Each block multiplies blockDim.x by tileK with tileK by blockDim.y
      // and accumulates the results into T0 Each thread multiplies 1 x tileK
      // with tileX by 1 and accumulates the results into T0
      for (int k = 0; k < tileK; ++k) {
        // reverse map T2's local row coordinate to global row coordinate
        // local: k, global: t * tileK + k
        const auto row = kOffset + k;
        if (i + row < n0) {
          t0_thread += t1_s[threadIdx.x * tileK + k] * t2[(i + row) + j * n2];
        }
      }
    }

    t0[i * n1 + j] += t0_thread;
    barrier.arrive_and_wait();
  }
}

void run(int deviceId, Strategy strategy, int tileK) {

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
#if DEBUG
  dim3 threads(16, 2, 1);
#else
  dim3 threads(kBlockDimX, kMaxBlockDim / kBlockDimX, 1);
#endif
  dim3 blocks(n0 / threads.x, n1 / threads.y, 1);
  fillMatrices(T0, T1, T2, blocks, threads, deviceId);

  std::cout << "Running with " << blocks.x << " x " << blocks.y << " blocks of "
            << threads.x << " x " << threads.y << " threads, K tile of "
            << tileK << std::endl;

  // hold tiles of T1 in shared memory
  uint32_t smemSize = threads.x * tileK * sizeof(float);

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
  case Strategy::AsynchronousCopyBarriers:
    bandedMatMul_asyncCopyBarriers<<<blocks, threads, smemSize>>>(
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

    // Try different block sizes and tile sizes
    std::cout << "GridDim,BlockDim,FLOPS,GFLOPS" << std::endl;

    for (uint32_t blockDim = kBlockDimX; blockDim <= kBlockDimXMax;
         blockDim += kBlockDimXStep) {

      threads.x = blockDim;
      threads.y = kMaxBlockDim / blockDim;
      blocks.x = n0 / threads.x;
      blocks.y = n1 / threads.y;
      smemSize = threads.x * tileK * sizeof(float);

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
          case Strategy::AsynchronousCopyBarriers:
            bandedMatMul_asyncCopyBarriers<<<blocks, threads, smemSize>>>(
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

      // BUGBUG: hangs for other block sizes
      if (strategy == Strategy::AsynchronousCopyBarriers) {
        break;
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
  Strategy strategy = Strategy::AsynchronousCopyBarriers;

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
  std::cout << "Using strategy " << argv[2] << std::endl;

  for (uint32_t tileK = 16; tileK <= 128; tileK *= 2) {
    run(deviceId, strategy, tileK);
  }

  return 0;
}