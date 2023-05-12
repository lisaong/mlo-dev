#include "utils.h"
#include <cuda_runtime.h>

__global__ void bandedMatMul(const Matrix &t0, const BandedMatrix &t1,
                             Matrix &t2) {

  float sum = t0.data[0];
}

void run(int nBand) {
  const int n0 = 1024;
  const int n1 = 1024;
  const int n2 = nBand;

  Matrix T0(n0, n1);
  BandedMatrix T1(n1, n2);
  Matrix T2(T1.width(), n1);

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

  T0.init(3);
  T1.init(4);
  T2.init(0);

  // Launch the kernel
  dim3 threads(16, 16, 1);
  dim3 blocks(n0 / threads.x, n1 / threads.y, 1);

  bandedMatMul<<<blocks, threads>>>(T0, T1, T2);

  CHECK(cudaDeviceSynchronize());

  cudaFree(T0.data);
  cudaFree(T1.data);
  cudaFree(T2.data);
}

int main(int argc, const char **argv) {
  int deviceId;
  CHECK(cudaGetDevice(&deviceId));
  run(16);
  return 0;
}