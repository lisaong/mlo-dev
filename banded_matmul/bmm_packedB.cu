// Packed B implementation of banded matrix multiplication
// where the B matrix is packed to improve locality:
//
// [ x 0 0 0 0 0 0 0 ]      [ x x x x x x x x ]
// [ x x 0 0 0 0 0 0 ]  =>  [ x x x x x x x 0 ]  B[1, 7] = 0
// [ x x x 0 0 0 0 0 ]      [ x x x x x x 0 0 ]  B[2, 6] = B[2, 7] = 0
// [ 0 x x x 0 0 0 0 ]
// [ 0 0 x x x 0 0 0 ]
// [ 0 0 0 x x x 0 0 ]
// [ 0 0 0 0 x x x 0 ]
// [ 0 0 0 0 0 x x x ]
//
// This assumes the A-matrix is banded:
//
// [ x x x 0 0 0 0 0 ]      [ x x x ]
// [ 0 x x x 0 0 0 0 ]      [ x x x ]
// [ 0 0 x x x 0 0 0 ]      [ x x x ]
// [ 0 0 0 x x x 0 0 ]  =>  [ x x x ]
// [ 0 0 0 0 x x x 0 ]      [ x x x ]
// [ 0 0 0 0 0 x x x ]      [ x x x ]
// [ 0 0 0 0 0 0 x x ]      [ x x 0 ]  A[6, 2] = 0
// [ 0 0 0 0 0 0 0 x ]      [ x 0 0 ]  A[7, 1] = A[7, 2] = 0

#include <assert.h>
#include <cstdint>

#include "utils.h"
#include <cuda_runtime.h>

// #define PREFETCH 1 // doesn't help
#define DEVICE_INIT 1

#define DEBUG 1
#if DEBUG
constexpr uint32_t N = 16;
#else
constexpr uint32_t N = 1024;
#endif // DEBUG

constexpr uint32_t kBandDim = N;
constexpr uint32_t kBlockDim = 16;
constexpr uint32_t kMaxBlockDim = 1024;
constexpr uint32_t kNumberOfOps = 2 * N * N * N;
constexpr uint32_t kMillisecondsInSeconds = 1000;
constexpr uint32_t kTimeLimit = 10 * kMillisecondsInSeconds;

__global__ void initWith(float num, float *a, int rows, int columns) {

  int i, j;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < columns;
         j += blockDim.y * gridDim.y) {
      a[i * columns + j] = num;
    }
  }
}

__global__ void initBandedWith(float num, float *a, int rows, int columns,
                               int band) {

  int i, j;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < band;
         j += blockDim.y * gridDim.y) {

      if ((i + j) < columns) {
        a[i * band + j] = num;
      } else {
        // zero out the lower right triangle
        a[i * band + j] = 0;
      }
    }
  }
}

__global__ void initTransposeBandedWith(float num, float *a, int rows,
                                        int columns, int band) {

  int i, j;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < band;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < columns;
         j += blockDim.y * gridDim.y) {

      if ((i + j) < rows) {
        a[i * columns + j] = num;
      } else {
        // zero out the lower right triangle
        a[i * columns + j] = 0;
      }
    }
  }
}

__global__ void bandedMatMul_PackedB(int n0, int n1, int n2, float *t0,
                                     const float *t1, const float *t2) {

  int i, j;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < n0;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < n1;
         j += blockDim.y * gridDim.y) {

      for (int k = 0; k < n2; ++k) {
        t0[i * n1 + j] += t1[i * n2 + k] * t2[k * n1 + j];
      }
    }
  }
}

bool checkCorrectness(int n0, int n1, int n2, const Matrix &T0,
                      const BandedMatrix &T1,
                      const TransposedBandedMatrix &T2) {
  Matrix T0_CPU(n0, n1);
  Matrix T2_CPU(T2.rows(), T2.columns());

  T0_CPU.data = reinterpret_cast<float *>(malloc(T0_CPU.size()));
  T0_CPU.init(0.0f);
  T2_CPU.data = reinterpret_cast<float *>(malloc(T2_CPU.size()));
  T2_CPU.init(33.0f);

  bandedMatMul_CPU(n0, n1, n2, T0_CPU.data, T1.data, T2_CPU.data);

#if DEBUG
  std::cout << "T0_CPU: " << std::endl;
  T0_CPU.print();
  std::cout << "T0: " << std::endl;
  T0.print();
#endif // DEBUG

  bool result = T0_CPU == T0;
  if (result) {
    std::cout << "Values match" << std::endl;
  } else {
    std::cerr << "Values do not match" << std::endl;
  }

  free(T0_CPU.data);
  free(T2_CPU.data);
  return result;
}

bool verify() {

  // n0: number of rows in T0 and T1
  // n1: number of columns in T0 and T2
  // n2: inner or shared dimension, i.e.
  //  number of columns in T1 and number of rows in T2

  const int n0 = N;
  const int n1 = N;
  const int n2 = kBandDim;

  Matrix T0(n0, n1);                               // output
  BandedMatrix T1(n0, n1, n2);                     // input
  TransposedBandedMatrix T2(T1.columns(), n1, n2); // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

  dim3 threads(kBlockDim, kBlockDim, 1);
  dim3 blocks(n0 / threads.x, n1 / threads.y, 1);

#if DEVICE_INIT
  initWith<<<blocks, threads>>>(0.0f, T0.data, T0.rows(), T0.columns());
  initBandedWith<<<blocks, threads>>>(22.0f, T1.data, T1.rows(), T1.columns(),
                                      T1.band());
  initTransposeBandedWith<<<blocks, threads>>>(33.0f, T2.data, T2.rows(),
                                               T2.columns(), T2.band());
  CHECK(cudaDeviceSynchronize());

#if DEBUG
  std::cout << "T1: " << std::endl;
  T1.print();
  std::cout << "T2: " << std::endl;
  T2.print();
#endif

#else
  T0.init(0.0f);
  T1.init(22.0f);
  T2.init(33.0f);
#endif

  bandedMatMul_PackedB<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
                                            T2.data);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  cudaMemPrefetchAsync(T0.data, T0.size(), cudaCpuDeviceId);
  bool result = checkCorrectness(n0, n1, n2, T0, T1, T2);

  cudaFree(T0.data);
  cudaFree(T1.data);
  cudaFree(T2.data);

  return result;
}

void benchmark(int deviceId) {
  // Runs the function until 10 seconds has elapsed

  cudaEvent_t _start;
  cudaEvent_t _stop;
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);

  const int n0 = N;
  const int n1 = N;
  const int n2 = kBandDim;

  Matrix T0(n0, n1);                               // output
  BandedMatrix T1(n0, n1, n2);                     // input
  TransposedBandedMatrix T2(T1.columns(), n1, n2); // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

#if DEVICE_INIT

  dim3 threadsInit(kBlockDim, kBlockDim, 1);
  dim3 blocksInit(n0 / threadsInit.x, n1 / threadsInit.y, 1);

  initWith<<<threadsInit, threadsInit>>>(0.0f, T0.data, T0.rows(),
                                         T0.columns());
  initBandedWith<<<threadsInit, threadsInit>>>(22.0f, T1.data, T1.rows(),
                                               T1.columns(), T1.band());
  initTransposeBandedWith<<<threadsInit, threadsInit>>>(
      33.0f, T2.data, T2.rows(), T2.columns(), T2.band());
  CHECK(cudaDeviceSynchronize());
#else
  T0.init(0.0f);
  T1.init(22.0f);
  T2.init(33.0f);
#endif // DEVICE_INIT

#if PREFETCH
  cudaMemPrefetchAsync(T0.data, T0.size(), deviceId);
  cudaMemPrefetchAsync(T1.data, T1.size(), deviceId);
  cudaMemPrefetchAsync(T2.data, T2.size(), deviceId);
#endif // PREFETCH

  for (uint32_t blockDim = kBlockDim; blockDim <= kMaxBlockDim;
       blockDim += kBlockDim) {

    dim3 threads(blockDim, blockDim, 1);
    dim3 blocks(n0 / threads.x, n1 / threads.y, 1);

    double elapsedTimeMilliseconds = 0.0f;
    uint64_t iterations = 0;
    float duration = 0.0f;

    cudaEventRecord(_start);
    while (elapsedTimeMilliseconds < kTimeLimit) {
      bandedMatMul_PackedB<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
                                                T2.data);
      cudaDeviceSynchronize();
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
  }

  cudaFree(T0.data);
  cudaFree(T1.data);
  cudaFree(T2.data);

  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
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

  if (verify()) {
    benchmark(deviceId);
  }
  return 0;
}