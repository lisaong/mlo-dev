#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>

#include "inc/timed_region.h"

using float16_t = _Float16;

#include "inc/ulp.h"

#ifndef HIP_ASSERT
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif

#define CDIV(n, block) (n + block - 1) / block

enum class Strategy
{
    Naive = 0,
    Tiled = 1
};

template <typename T>
__global__ void init(T *a, uint64_t M, uint64_t N)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    for (; i < M; i += blockDim.y * gridDim.y)
    {
        for (; j < N; j += blockDim.x * gridDim.x)
        {
            a[i * N + j] = static_cast<T>(i * N + j) / static_cast<T>(N * M / 4);
        }
    }
}

// cf. https://gitlab.com/syifan/hipbookexample/-/blob/main/Chapter5/MatrixMultiplication/main.cpp

__global__ void matrixMultiplyTiled(float16_t *A, float16_t *B, float *C, uint64_t M, uint64_t N, uint64_t K, int tileSize)
{
    // C[i, j] += A[i, k] * B[k, j]
    // (M, N)    (M, K)    (K, N)
    //   where y => rows (i), x => colummns (j)

    extern __shared__ float subTileA[];
    float *subTileB = &subTileA[tileSize * tileSize];

    // cumulative sum across the full K dimension
    float sum = 0.0f;

    // load the A and B tiles
    const int row = blockIdx.y * tileSize + threadIdx.y;
    const int col = blockIdx.x * tileSize + threadIdx.x;
    const int numTiles = CDIV(K, tileSize);

    // walk the K dimension in tiles
    for (int k = 0; k < numTiles; ++k)
    {
        // load tileSize rows of A (i dimension, threadIdx.y)
        // and tileSize columns (k dimension, threadIdx.x)
        int aRow = row;
        int aCol = k * tileSize + threadIdx.x;
        int elem = threadIdx.y * tileSize + threadIdx.x;

        if (aRow < M && aCol < K)
        {
            // only tileSize x tileSize will be copied per workgroup
            subTileA[elem] = A[aRow * K + aCol];
        }
        else
        {
            subTileA[elem] = 0.0f;
        }

        // load tileSize rows of B (k dimension, threadIdx.y)
        // and tileSize cols of B (j dimension, threadIdx.x)
        int bRow = k * tileSize + threadIdx.y;
        int bCol = col;

        if (bRow < K && bCol < N)
        {
            // only tileSize x tileSize will be copied per workgroup
            subTileB[elem] = B[bRow * N + bCol];
        }
        else
        {
            subTileB[elem] = 0.0f;
        }

        __syncthreads(); // wait for complete tile to be loaded

        // multiply subTileA with subTileB
        // each thread will take the threadIdx.y's row across the kk dimension
        // and multiply that by threadIdx.x's column across the kk dimension
        float tileSum = 0.0f; // for clarity
        for (int kk = 0; kk < tileSize; ++kk)
        {
            if (k * tileSize + kk < K)
            {
                tileSum += subTileA[threadIdx.y * tileSize + kk] * subTileB[kk * tileSize + threadIdx.x];
            }
        }
        // aggregate the tile-local sum into the k sum
        sum += tileSum;

        __syncthreads(); // wait for processing of the current tile to be complete, otherwise
                         // other threads may update subTileA or subTileB before we are done
                         // with computing all the thread-local sums

        // update the result
        C[row * N + col] = sum;
    }
}

__global__ void matrixMultiplyNaive(float16_t *A, float16_t *B, float *C, uint64_t M, uint64_t N, uint64_t K)
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

#ifdef VERIFY
void matrixMultiplyCPU(float16_t *A, float16_t *B, float *C, uint64_t M, uint64_t N, uint64_t K)
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
#endif // VERIFY

int run(int deviceId, int tileSize, Strategy strategy)
{
#ifdef VERIFY
    constexpr uint64_t M = 64;
#else
    constexpr uint64_t M = 2 << 12;
#endif // VERIFY
    constexpr uint64_t N = M;
    constexpr uint64_t K = M;

    const dim3 numThreads(tileSize, tileSize, 1);
    const dim3 numBlocks(CDIV(M, numThreads.x), CDIV(N, numThreads.y), 1);

    // alloc
    float16_t *d_a;
    float16_t *d_b;
    float *d_c;
    HIP_ASSERT(hipMallocManaged(&d_a, M * K * sizeof(float16_t)));
    HIP_ASSERT(hipMallocManaged(&d_b, K * N * sizeof(float16_t)));
    HIP_ASSERT(hipMallocManaged(&d_c, M * N * sizeof(float)));

    HIP_ASSERT(hipMemPrefetchAsync(d_a, M * K * sizeof(float16_t), deviceId));
    HIP_ASSERT(hipMemPrefetchAsync(d_b, K * N * sizeof(float16_t), deviceId));
    HIP_ASSERT(hipMemPrefetchAsync(d_c, M * N * sizeof(float), deviceId));

    init<<<numBlocks, numThreads>>>(d_a, M, K);
    init<<<numBlocks, numThreads>>>(d_b, K, N);
    hipDeviceSynchronize();

    std::stringstream ss;
    ss << numBlocks.x << "," << numThreads.x; // BUGBUG: assumes square sizes
    if (strategy == Strategy::Naive)
    {
        TimedRegion r(ss.str());

        matrixMultiplyNaive<<<numBlocks, numThreads>>>(d_a, d_b, d_c, M, N, K);
        hipDeviceSynchronize();
    }
    else
    {
        int sharedMemorySize = tileSize * tileSize * sizeof(float) * 2; // subTileA and subTileB

        TimedRegion r(ss.str());

        matrixMultiplyTiled<<<numBlocks, numThreads, sharedMemorySize>>>(d_a, d_b, d_c, M, N, K, tileSize);
        hipDeviceSynchronize();
    }

#ifdef VERIFY
    {
        float *cVerify = new float[M * N];
        matrixMultiplyCPU(d_a, d_b, cVerify, M, N, K);

        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                auto ulpDiff = ULPDiff(d_c[i * N + j], cVerify[i * N + j]);
                if (ulpDiff > 1e4)
                {
                    std::cout << "Error: C[" << i << ", " << j << "] = "
                              << d_c[i * N + j] << ", expected "
                              << cVerify[i * N + j] << ", ulpdiff "
                              << ulpDiff << std::endl;
                    return -1;
                }
            }
        }

        delete[] cVerify;
    }
#endif // VERIFY

    HIP_ASSERT(hipFree(d_a));
    HIP_ASSERT(hipFree(d_b));
    HIP_ASSERT(hipFree(d_c));

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

    Strategy strategy = Strategy::Tiled;
    if (argc > 1)
    {
        strategy = static_cast<Strategy>(atoi(argv[1]));
    }

    std::cout << "grid_size,block_size,elapsed_msec" << std::endl;
    int result = 0;
    for (int numThreads = 32; numThreads <= 2500 && result == 0; numThreads += 32)
    {
        result = run(deviceId, numThreads, strategy);
    }
    return result;
}