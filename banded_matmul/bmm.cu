#include "utils.h"
#include <cuda_runtime.h>

void run(int nBand) {
  const int n0 = 1024;
  const int n1 = 1024;
  const int n2 = nBand;

  // dense size
  float *T0, *T1, *T2;
  const int n2Full = n0 + n2;

  CHECK(cudaMallocManaged(&T0, n0 * n1 * sizeof(float)));
  CHECK(cudaMallocManaged(&T1, n0 * n2Full * sizeof(float)));
  CHECK(cudaMallocManaged(&T2, n2Full * sizeof(float)));

  cudaFree(T0);
  cudaFree(T1);
  cudaFree(T2);
}

int main(int argc, const char **argv) {
  int deviceId;
  CHECK(cudaGetDevice(&deviceId));
  run(16);
  return 0;
}