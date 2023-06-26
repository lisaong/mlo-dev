#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>

#include "inc/timed_region.h"

using float16_t = _Float16;

#ifndef HIP_ASSERT
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif

template <typename T>
__global__ void init(T *a, int n)
{
    const int stride = blockDim.x * gridDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += stride)
    {
        a[i] = static_cast<T>(i) / static_cast<T>(1024);
    }
}

// cf. https://gitlab.com/syifan/hipbookexample/-/blob/main/Chapter5/MatrixMultiplication/main.cpp

__global__ void matrixMultiplyNaive(float16_t *A, float16_t *B, float *C, int M, int N, int K)
{
    // C[i, j] = A[i, k] * B[k, j]
    // (M, N)    (M, K)    (K, N)
    //   where y => rows (i), x => colummns (j)
    const int i = blockDim.y * blockIdx.y + threadIdx.y;
    const int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < M && j < N)
    {
        // multiply then sum along the k dimension
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            sum += A[i * K + k] * B[k * N + j];
        }
    }

    C[i * N + j] = sum;
}


int run(int deviceId, int numBlocks)
{
}