#include "utils.h"
#include <cuda_runtime.h>

void run(int nBand) {
  const int n0 = 1024;
  const int n1 = 1024;
  const int n2 = nBand;

  // dense size
  Matrix T0(n0, n1);
  BandedMatrix T1(n1, n2);
  Matrix T2(T1.width(), n1);

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

  initWith(T0, 3);
  initWith(T1, 4);
  initWith(T2, 0);

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