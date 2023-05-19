// Double-buffered banded matrix multiplication
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdint>
#include <cuda_runtime.h>

#define DEBUG 1
#include "constants.h"
#include "utils.h"

// https://developer.nvidia.com/blog/cooperative-groups/
namespace cg = cooperative_groups;

enum class Strategy { SynchronousCopy = 0, AsynchronousCopy = 1 };

__global__ void bandedMatMul_syncCopy(int n0, int n1, int n2, float *t0,
                                      const float *t1, const float *t2) {

  int i, j, k;

  auto cta = cg::this_thread_block();

  // load the t0 and t1 sub-matrices into shared memory
  extern __shared__ float t0_s[];
  float *t1_s = &t0_s[cta.size()];

  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < n0;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < n1;
         j += blockDim.y * gridDim.y) {
      t0_s[threadIdx.x * blockDim.y + threadIdx.y] = t0[i * n1 + j];
      t1_s[threadIdx.x * blockDim.y + threadIdx.y] = t1[i * n2 + j];

      cta.sync();

      // treat t2 as column major
      for (k = 0; k < n2 && (i + k) < n0; ++k) {
        t0_s[threadIdx.x * blockDim.y + threadIdx.y] +=
            t1_s[threadIdx.x * blockDim.y + threadIdx.y] * t2[(i + k) + j * n2];
      }
      cta.sync();

      // write back to global memory
      t0[i * n1 + j] = t0_s[threadIdx.x * blockDim.y + threadIdx.y];
    }
  }
}

__global__ void bandedMatMul_asyncCopy(int n0, int n1, int n2, float *t0,
                                       const float *t1, const float *t2) {

  extern __shared__ float t0_s[];

  // cf. MatrixMulAsyncCopySingleStage in
  // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu
  auto cta = cg::this_thread_block();
  float *t1_s = &t0_s[cta.size()];

  // cooperatively copy each blockDim.x * blockDim.y tile of t0 and t1 to shared
  // memory
  int numRows = blockDim.x;
  int columnOffset = cta.group_index().y * blockDim.y;
  int columnStride = blockDim.y;

  for (int b = 0; b < numRows; ++b) {
    // copy the row
    int rowOffset = cta.group_index().x * blockDim.x + b;
    cg::memcpy_async(cta, t0_s, &t0[rowOffset * n1 + columnOffset],
                     sizeof(float) * columnStride);
    cg::memcpy_async(cta, t1_s, &t1[rowOffset * n2 + columnOffset],
                     sizeof(float) * columnStride);
    cg::wait(cta);
  }

  // compute the tile
  int i = cta.group_index().x * blockDim.x + threadIdx.x;
  int j = cta.group_index().y * blockDim.y + threadIdx.y;
  for (int k = 0; k < n2 && (i + k) < n0; ++k) {
    t0_s[threadIdx.x * blockDim.y + threadIdx.y] +=
        t1_s[threadIdx.x * blockDim.y + threadIdx.y] * t2[(i + k) + j * n2];
  }
  cta.sync();

  for (int b = 0; b < numRows; ++b) {
    // write back to t0 global memory
    int rowOffset = cta.group_index().x * blockDim.x + b;
    cg::memcpy_async(cta, &t0[rowOffset * n1 + columnOffset], t0_s,
                     sizeof(float) * columnStride);
    cg::wait(cta);
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

  // Verify
  uint32_t smemSize;
  switch (strategy) {
  case Strategy::SynchronousCopy:
    // shared memory: [t0 sub-matrix, t1 sub-matrix]
    smemSize = threads.x * threads.y * sizeof(float) * 2;
    bandedMatMul_syncCopy<<<blocks, threads, smemSize>>>(n0, n1, n2, T0.data,
                                                         T1.data, T2.data);
    break;
  case Strategy::AsynchronousCopy:
    // shared memory: [t0 sub-matrix, t1 sub-matrix]
    smemSize = threads.x * threads.y * sizeof(float) * 2;
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

      try {
        double elapsedTimeMilliseconds = 0.0f;
        uint64_t iterations = 0;
        float duration = 0.0f;

        // Runs the function until 10 seconds has elapsed
        cudaEventRecord(_start);
        while (elapsedTimeMilliseconds < kTimelimit) {

          switch (strategy) {
          case Strategy::SynchronousCopy:
            smemSize = threads.x * threads.y * sizeof(float) * 2;
            bandedMatMul_syncCopy<<<blocks, threads, smemSize>>>(
                n0, n1, n2, T0.data, T1.data, T2.data);
            break;
          case Strategy::AsynchronousCopy:
            smemSize = threads.x * threads.y * sizeof(float) * 2;
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