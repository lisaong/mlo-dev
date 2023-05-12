#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


cudaError_t CHECK(cudaError_t res) {
  if (cudaSuccess != res) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
    exit(EXIT_FAILURE);
  }
  return res;
}


int main(int argc, const char **argv) {
  int deviceId;
  CHECK(cudaGetDevice(&deviceId));

  return 0;
}