#include <cstdint>

#include "utils.h"
#include <cuda_runtime.h>

#define DEBUG 0

constexpr uint32_t kBandDim = 3;
constexpr uint32_t kBlockDim = 16;
constexpr uint32_t N = 1024;
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

  T0.init(11);
  T1.init(22);
  T2.init(33);

  dim3 threads(kBlockDim, kBlockDim, 1);
  dim3 blocks(n0 / threads.x, n1 / threads.y, 1);

  bandedMatMul_Naive<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
                                          T2.data);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  bool result = checkCorrectness(n0, n1, n2, T0, T1, T2);

  cudaFree(T0.data);
  cudaFree(T1.data);
  cudaFree(T2.data);

  return result;
}

void benchmark() {
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

  T0.randomInit(11);
  T1.randomInit(22);
  T2.randomInit(33);

  dim3 threads(kBlockDim, kBlockDim, 1);
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
  std::cout << "Iterations: " << iterations << ", FLOPS: " << flops
            << std::endl;

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
    benchmark();
  }
  return 0;
}