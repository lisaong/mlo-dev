#pragma once

#include <stdio.h>

#include <cuda_runtime.h>

cudaError_t CHECK(cudaError_t res) {
  if (cudaSuccess != res) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(res));
    exit(EXIT_FAILURE);
  }
  return res;
}

void initWith(float num, float *a, int N) {
  for (int i = 0; i < N; ++i) {
    a[i] = num;
  }
}
