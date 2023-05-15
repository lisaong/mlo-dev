#include <cstdint>

#include "utils.h"
#include <cuda_runtime.h>

// #define PREFETCH 0 // doesn't help
#define DEVICE_INIT 1

// #define DEBUG 1
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
constexpr uint32_t kTimelimit = 10 * kMillisecondsInSeconds;

void bandedMatMul_CPU(int n0, int n1, int n2, float *t0, const float *t1,
                      const float *t2) {
  /*
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                t0[i, j] += t1[i, k] * t2[i + k, j]
  */
  int i, j, k;
  for (i = 0; i < n0; ++i) {
    for (j = 0; j < n1; ++j) {
      for (k = 0; k < n2 && (i + k) < n0; ++k) {
        t0[i * n1 + j] += t1[i * n2 + k] * t2[(i + k) * n1 + j];
      }
    }
  }
}

__global__ void initWith(float num, float *a, int N) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < N; i += stride) {
    a[i] = num;
  }
}

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

bool checkCorrectness(int n0, int n1, int n2, const Matrix &T0,
                      const BandedMatrix &T1, const Matrix &T2) {
  Matrix T0_CPU(n0, n1);
  T0_CPU.data = reinterpret_cast<float *>(malloc(T0_CPU.size()));
  T0_CPU.init(11);

  bandedMatMul_CPU(n0, n1, n2, T0_CPU.data, T1.data, T2.data);

#if DEBUG
  for (int i = 0; i < T0_CPU.numElements(); ++i) {
    std::cout << "CPU: " << T0_CPU.data[i] << ", Device: " << T0.data[i]
              << std::endl;
  }
#endif // DEBUG

  bool result = T0_CPU == T0;
  if (result) {
    std::cout << "Values match" << std::endl;
  } else {
    std::cerr << "Values do not match" << std::endl;
  }

  free(T0_CPU.data);
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

  Matrix T0(n0, n1);           // output
  BandedMatrix T1(n0, n1, n2); // input
  Matrix T2(T1.columns(), n1); // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

#if DEVICE_INIT
  initWith<<<T0.numElements() / kBlockDim, kBlockDim>>>(11.0f, T0.data,
                                                        T0.numElements());
  initWith<<<T1.numElements() / kBlockDim, kBlockDim>>>(22.0f, T1.data,
                                                        T1.numElements());
  initWith<<<T2.numElements() / kBlockDim, kBlockDim>>>(33.0f, T2.data,
                                                        T2.numElements());
  CHECK(cudaDeviceSynchronize());
#else
  T0.init(11);
  T1.init(22);
  T2.init(33);
#endif

  dim3 threads(kBlockDim, kBlockDim, 1);
  dim3 blocks(n0 / threads.x, n1 / threads.y, 1);

  bandedMatMul_Naive<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
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

  Matrix T0(n0, n1);           // output
  BandedMatrix T1(n0, n1, n2); // input
  Matrix T2(T1.columns(), n1); // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

#if DEVICE_INIT
  initWith<<<T0.numElements() / kBlockDim, kBlockDim>>>(11.0f, T0.data,
                                                        T0.numElements());
  initWith<<<T1.numElements() / kBlockDim, kBlockDim>>>(22.0f, T1.data,
                                                        T1.numElements());
  initWith<<<T2.numElements() / kBlockDim, kBlockDim>>>(33.0f, T2.data,
                                                        T2.numElements());
  CHECK(cudaDeviceSynchronize());
#else
  T0.init(11.0f);
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
    while (elapsedTimeMilliseconds < kTimelimit) {
      bandedMatMul_Naive<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
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