#include "utils.h"
#include <cuda_runtime.h>

#define DEBUG 0

constexpr int kBandDim = 3;
constexpr int kBlockDim = 16;
constexpr int N = 1024;

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
      for (k = 0; k < n2; ++k) {
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
      for (k = 0; k < n2; ++k) {
        t0[i * n1 + j] += t1[i * n2 + k] * t2[(i + k) * n1 + j];
      }
    }
  }
}

void run() {

  // n0: number of rows in T0 and T1
  // n1: number of columns in T0 and T2
  // n2: inner or shared dimension, i.e.
  //  number of columns in T1 and number of rows in T2

  const int n0 = N;
  const int n1 = N;
  const int n2 = kBandDim;

  Matrix T0(n0, n1);           // output
  BandedMatrix T1(n0, n2);     // input
  Matrix T2(T1.columns(), n1); // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

  T0.init(1);
  T1.init(2);
  T2.init(3);

  dim3 threads(kBlockDim, kBlockDim, 1);
  dim3 blocks(n0 / threads.x, n1 / threads.y, 1);

  bandedMatMul_Naive<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
                                          T2.data);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  Matrix T0_CPU(n0, n1);
  T0_CPU.data = reinterpret_cast<float *>(malloc(T0_CPU.size()));
  T0_CPU.init(1);

  bandedMatMul_CPU(n0, n1, n2, T0_CPU.data, T1.data, T2.data);

#if DEBUG
  for (int i = 0; i < T0.size(); ++i) {
    std::cout << "CPU: " << T0_CPU.data[i] << ", Device: " << T0.data[i]
              << std::endl;
  }
#endif // DEBUG

  if (T0_CPU != T0) {
    std::cerr << "Values do not match" << std::endl;
  } else {
    std::cout << "Values match" << std::endl;
  }

  cudaFree(T0.data);
  cudaFree(T1.data);
  cudaFree(T2.data);
  free(T0_CPU.data);
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

  run();
  return 0;
}