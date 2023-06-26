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
    // C[i, j] += A[i, k] * B[k, j]
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
        C[i * N + j] = sum;
    }
}

void matrixMultiplyCPU(float16_t *A, float16_t *B, float *C, int M, int N, int K)
{
    // C[i, j] += A[i, k] * B[k, j]
    // (M, N)    (M, K)    (K, N)
    //   where y => rows (i), x => colummns (j)
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int run(int deviceId, int numBlocks)
{
    constexpr int numThreads = 256;
    constexpr int M = 8192;
    constexpr int N = M;
    constexpr int K = M;

    float16_t *d_a;
    float16_t *d_b;
    float *d_cNaive;
    float *d_cTiled;
    HIP_ASSERT(hipMallocManaged(&d_a, M * K * sizeof(float16_t)));
    HIP_ASSERT(hipMallocManaged(&d_b, K * N * sizeof(float16_t)));
    HIP_ASSERT(hipMallocManaged(&d_cNaive, M * N * sizeof(float)));

    HIP_ASSERT(hipMemPrefetchAsync(d_a, M * K * sizeof(float16_t), deviceId));
    HIP_ASSERT(hipMemPrefetchAsync(d_b, K * N * sizeof(float16_t), deviceId));
    HIP_ASSERT(hipMemPrefetchAsync(d_cNaive, M * N * sizeof(float), deviceId));

    init<<<numBlocks, numThreads>>>(d_a, M * K);
    init<<<numBlocks, numThreads>>>(d_b, K * N);

    {
        std::stringstream ss;
        ss << numBlocks << "," << numThreads;

        TimedRegion r(ss.str());

        matrixMultiplyNaive<<<numBlocks, numThreads>>>(d_a, d_b, d_cNaive, M, N, K);
    }

    // Verify
    float *cVerify = new float[M * N];
    matrixMultiplyCPU(d_a, d_b, cVerify, M, N, K);
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (abs(d_cNaive[i * N + j] - cVerify[i * N + j]) > 1e-5)
            {
                std::cout << "Error: C[" << i << ", " << j << "] = "
                          << d_cNaive[i * N + j] << ", expected "
                          << cVerify[i * N + j] << std::endl;
                return -1;
            }
        }
    }

    delete[] cVerify;
    HIP_ASSERT(hipFree(d_a));
    HIP_ASSERT(hipFree(d_b));
    HIP_ASSERT(hipFree(d_cNaive));
    HIP_ASSERT(hipFree(d_cTiled));

    return 0;
}

int main(int argc, const char **argv)
{
    int deviceId = 0;
    HIP_ASSERT(hipGetDevice(&deviceId));

    int supportsManagedMemory = 0;
    HIP_ASSERT(hipDeviceGetAttribute(&supportsManagedMemory,
                                     hipDeviceAttributeManagedMemory, deviceId));

    if (supportsManagedMemory == 0)
    {
        std::cout << "Managed memory is not supported for device " << deviceId << std::endl;
        return -1;
    }

    std::cout << "grid_size,block_size,elapsed_msec" << std::endl;
    int result = 0;
    for (int numBlocks = 32; numBlocks <= 4096 && result == 0; numBlocks += 32)
    {
        result = run(deviceId, numBlocks);
    }
    return result;
}