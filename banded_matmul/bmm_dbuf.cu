// Double-buffered banded matrix multiplication
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
// #include <cuda/pipeline>
#include <cuda_runtime.h>

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

  assert(n0 == cta.num_threads());
  assert(n2 % blockDim.x == 0);

  // T0: prepare the result tile
  // Each block copies blockDim.x x blockDim.y entries, one entry per thread
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  t0_s[threadIdx.x * blockDim.y + threadIdx.y] = t0[i * n1 + j];

  // Due to shared memory limitations, we cannot fit complete rows or columns of
  // T1 and T2 per block:
  //   T1: only blockDim.x by tileK in shared memory
  //   T2: only tileK by blockDim.y in shared memory
  // Perform the copying and multiplication per tile, then accumulate
  // the results in a blockDim.x by blockDim.y T0 tile
  const auto numTilesPerBlock = n1 / tileK;
  const auto colsPerThread = tileK / blockDim.y;
  const auto rowsPerThread = tileK / blockDim.x;

  // Loop through each tile of T1 and T2
  for (int t = 0; t < numTilesPerBlock; ++t) {
    // T1: Each block copies a blockDim.x by tileK tile per iteration
    // Each thread copies a 1 by (tileK / blockDim.y) row, sliding column-wise
    // by colsPerThread
    const auto t1GlobalX = blockIdx.x * blockDim.x;
    const auto t1GlobalY = t * tileK;
    const auto t1ThreadX = threadIdx.x;

    for (int k = 0; k < colsPerThread; ++k) {
      const auto t1ThreadY = threadIdx.y * colsPerThread + k;
      const auto row = t1GlobalX + t1ThreadX;
      const auto col = t1GlobalY + t1ThreadY;
      const auto idx = row * n1 + col;
      const auto sIdx = t1ThreadX * tileK + t1ThreadY;
      t1_s[sIdx] = t1[idx];
    }

#if T2_SMEM
    // T2: Each block copies a tileK by blockDim.y tile;
    // Each thread copies a tileK / blockDim.x by 1 column
    // T2 values need to be shifted down by the row index of T1
    const auto t2GlobalY = blockIdx.y * blockDim.y;
    const auto t2GlobalX = t * tileK;
    const auto t2ThreadY = threadIdx.y;
    const auto shiftOffset = t1GlobalX + t1ThreadX;

    for (int k = 0; k < rowsPerThread; ++k) {
      const auto t2ThreadX = threadIdx.x * rowsPerThread + k;
      const auto row = shiftOffset + t2GlobalX + t2ThreadX;
      if (row < n0) {
        const auto col = t2GlobalY + t2ThreadY;
        // column major layout
        const auto idx = row + col * n1;
        const auto sIdx = t2ThreadX + t2ThreadY * tileK;
        t2_s[sIdx] = t2[idx];
      }
    }
#endif // T2_SMEM
    cta.sync();

    // Each block multiplies blockDim.x by tileK with tileK by blockDim.y and
    // accumulates the results into T0
    // Each thread multiplies 1 x tileK with tileX by 1 and
    // accumulates the results into T0
    const auto sIdx = threadIdx.x * blockDim.y + threadIdx.y;
    for (int k = 0; k < tileK; ++k) {
#if T2_SMEM
      t0_s[sIdx] +=
          t1_s[threadIdx.x * tileK + k] * t2_s[threadIdx.y * tileK + k];
#else
      t0_s[sIdx] +=
          t1_s[threadIdx.x * tileK + k] * t2[(i + t * tileK + k) * n1 + j];
#endif // T2_SMEM
    }
  }

  t0[i * n1 + j] = t0_s[threadIdx.x * blockDim.y + threadIdx.y];
}

__global__ void bandedMatMul_asyncCopy(int n0, int n1, int n2, float *t0,
                                       const float *t1, const float *t2) {

  int i, j, k;

  auto cta = cg::this_thread_block();

  extern __shared__ float t0_s[];
  float *t1_s = &t0_s[cta.size()];

  // each block will copy n0 / gridDim.x rows
  const auto numRows = n0 / gridDim.x;
  const auto rowOffset = blockIdx.x * numRows;
  for (int r = 0; r < numRows; ++r) {
    // copy a row of t0 and t1 into shared memory
    i = rowOffset + r;
    cg::memcpy_async(cta, t0_s, &t0[i * n1], sizeof(float) * cta.size());
    cg::memcpy_async(cta, t1_s, &t1[i * n1], sizeof(float) * cta.size());

    cg::wait(cta); // wait for copies to complete

    // compute the row, assumes the number of threads == row width
    j = threadIdx.x * blockDim.y + threadIdx.y;

    // treat t2 as column major
    for (k = 0; i + k < n1; ++k) {
      t0_s[j] += t1_s[j] * t2[(i + k) + j * n2];
    }
    cta.sync(); // wait for all threads to compute

    // write back to global memory
    t0[i * n1 + j] = t0_s[j];
    cta.sync(); // wait for all threads to consume
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
    bandedMatMul_asyncCopy<<<blocks, threads, smemSize>>>(n0, n1, n2, T0.data,
                                                          T1.data, T2.data);
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
      blocks.x = ceildiv(n0, threads.x);
      blocks.y = ceildiv(n1, threads.y);
      tileK = n1 / threads.y; // TODO: check
      smemSize = threads.x * threads.y * sizeof(float) +
                 threads.x * n2 * sizeof(float) +
                 n2 * threads.y * sizeof(float);

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
                n0, n1, n2, T0.data, T1.data, T2.data);
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