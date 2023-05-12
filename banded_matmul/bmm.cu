#include "utils.h"
#include <cuda_runtime.h>

void bandedMatMul_CPU(int n0, int n1, int n2, float *t0, const float *t1, const float *t2) {
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

__global__ void bandedMatMul_Naive(int n0, int n1, int n2, float *t0, const float *t1,
                                   const float *t2) {

  int i, j, k;
  for (i = blockIdx.x * blockDim.x + threadIdx.x; i < n0;
       i += blockDim.x * gridDim.x) {
    for (j = blockIdx.y * blockDim.y + threadIdx.y; j < n1;
         j += blockDim.x * gridDim.x) {
      for (k = blockIdx.z * blockDim.z + threadIdx.z; k < n2;
           k += blockDim.z * gridDim.z) {
        t0[i * n1 + j] += t1[i * n2 + k] * t2[(i + k) * n1 + j];
      }
    }
  }
}

void run(int nBand) {
  const int n0 = 1024;
  const int n1 = 1024;
  const int n2 = nBand;

  Matrix T0(n0, n1);           // output
  BandedMatrix T1(n1, n2);     // input
  Matrix T2(T1.columns(), n1); // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

  T0.init(3);
  T1.init(4);
  T2.init(0);

  dim3 threads(16, 16, 16);
  dim3 blocks(n0 / threads.x, n1 / threads.y, n2 / threads.z);

  bandedMatMul_Naive<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
                                          T2.data);

  CHECK(cudaDeviceSynchronize());

  Matrix T0_CPU(n0, n1);
  T0_CPU.data = reinterpret_cast<float*>(malloc(T0_CPU.size()));
  T0_CPU.init(3);

  bandedMatMul_CPU(n0, n1, n2, T0_CPU.data, T1.data, T2.data);

  cudaFree(T0.data);
  cudaFree(T1.data);
  cudaFree(T2.data);
  free(T0_CPU.data);
}

int main(int argc, const char **argv) {
  int deviceId;
  CHECK(cudaGetDevice(&deviceId));
  run(16);
  return 0;
}