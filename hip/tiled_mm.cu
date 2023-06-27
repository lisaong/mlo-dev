#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>

#include "inc/assert.h"
#include "inc/timed_region.h"

using float16_t = _Float16;

#include "inc/ulp.h"

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

void matrixMultiplyCPU(float16_t *A, float16_t *B, float *C, uint64_t M, uint64_t N, uint64_t K, bool managedMemory)
{
    float16_t *a;
    float16_t *b;

    if (managedMemory)
    {
        a = A;
        b = B;
    }
    else
    {
        a = new float16_t[M * K];
        b = new float16_t[K * N];

        HIP_ASSERT(hipMemcpy(a, A, M * K * sizeof(float16_t), hipMemcpyDeviceToHost));
        HIP_ASSERT(hipMemcpy(b, B, K * N * sizeof(float16_t), hipMemcpyDeviceToHost));
    }

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
                sum += a[i * K + k] * b[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    if (!managedMemory)
    {
        delete[] a;
        delete[] b;
    }
}

template <typename TIn, typename TOut>
int init(int deviceId, TIn **input1, TIn **input2, TOut **output, int M, int N, int K, bool managedMemory)
{
    if (managedMemory)
    {
        HIP_ASSERT(hipMallocManaged(input1, M * K * sizeof(TIn)));
        HIP_ASSERT(hipMallocManaged(input2, K * N * sizeof(TIn)));
        HIP_ASSERT(hipMallocManaged(output, M * N * sizeof(TOut)));

        HIP_ASSERT(hipMemPrefetchAsync(*input1, M * K * sizeof(TIn), deviceId));
        HIP_ASSERT(hipMemPrefetchAsync(*input2, K * N * sizeof(TIn), deviceId));
        HIP_ASSERT(hipMemPrefetchAsync(*output, M * N * sizeof(TOut), deviceId));
    }
    else
    {
        HIP_ASSERT(hipMalloc(input1, M * K * sizeof(TIn)));
        HIP_ASSERT(hipMalloc(input2, K * N * sizeof(TIn)));
        HIP_ASSERT(hipMalloc(output, M * N * sizeof(TOut)));
    }

    const dim3 numThreads(256, 256, 1);
    dim3 numBlocks(CDIV(M, numThreads.x), CDIV(K, numThreads.y), 1);
    init<<<numBlocks, numThreads>>>(*input1, M, K);

    numBlocks.x = CDIV(K, numThreads.x);
    numBlocks.y = CDIV(N, numThreads.y);
    init<<<numBlocks, numThreads>>>(*input2, K, N);
    hipDeviceSynchronize();
    return 0;
}

template <typename TIn, typename TOut>
void cleanup(TIn *input1, TIn *input2, TOut *output)
{
    HIP_ASSERT(hipFree(input1));
    HIP_ASSERT(hipFree(input2));
    HIP_ASSERT(hipFree(output));
}

template <typename TIn, typename TOut>
int run(int deviceId, TIn *d_a, TIn *d_b, TOut *d_c, TOut *verify, int M, int N, int K, int tileSize, Strategy strategy, bool managedMemory)
{
    const dim3 numThreads(tileSize, tileSize, 1);
    const dim3 numBlocks(CDIV(M, numThreads.x), CDIV(N, numThreads.y), 1);

    std::stringstream ss;
    ss << numBlocks.x << "," << numThreads.x; // BUGBUG: assumes square sizes
    if (strategy == Strategy::Naive)
    {
        TimedRegion r(ss.str());

        matrixMultiplyNaive<<<numBlocks, numThreads>>>(d_a, d_b, d_c, M, N, K);
        HIP_ASSERT(hipGetLastError());
        HIP_ASSERT(hipDeviceSynchronize());
    }
    else
    {
        int sharedMemorySize = tileSize * tileSize * sizeof(TOut) * 2; // subTileA and subTileB

        TimedRegion r(ss.str());

        matrixMultiplyTiled<<<numBlocks, numThreads, sharedMemorySize>>>(d_a, d_b, d_c, M, N, K, tileSize);
        HIP_ASSERT(hipGetLastError());
        HIP_ASSERT(hipDeviceSynchronize());
    }

    if (verify != nullptr)
    {
        TOut *c = nullptr;
        if (managedMemory)
        {
            c = d_c;
        }
        else
        {
            c = new TOut[M * N];
            HIP_ASSERT(hipMemcpy(c, d_c, M * N * sizeof(TOut), hipMemcpyDeviceToHost));
        }

        bool match = true;
        for (int i = 0; i < M && match; ++i)
        {
            for (int j = 0; j < N && match; ++j)
            {
                auto ulpDiff = ULPDiff(c[i * N + j], verify[i * N + j]);
                if (ulpDiff > 1e4)
                {
                    std::cout << "Error: C[" << i << ", " << j << "] = "
                              << c[i * N + j] << ", expected "
                              << verify[i * N + j] << ", ulpdiff "
                              << ulpDiff << std::endl;
                    match = false;
                }
            }
        }

        if (!managedMemory)
        {
            delete[] c;
        }

        if (!match)
        {
            return -1;
        }
    }

    return 0;
}

int main(int argc, const char **argv)
{
    int deviceId = 0;
    HIP_ASSERT(hipGetDevice(&deviceId));

    int supportsManagedMemory = 0;
    HIP_ASSERT(hipDeviceGetAttribute(&supportsManagedMemory,
                                     hipDeviceAttributeManagedMemory, deviceId));

    Strategy strategy = Strategy::Tiled;
    if (argc > 1)
    {
        strategy = static_cast<Strategy>(atoi(argv[1]));
    }

#ifdef VERIFY
    constexpr uint64_t M = 64;
#else
    constexpr uint64_t M = 2 << 14;
#endif // VERIFY
    constexpr uint64_t N = M;
    constexpr uint64_t K = M;

    float16_t *d_a;
    float16_t *d_b;
    float *d_c;
    float *cVerify = nullptr;

    int result = init(deviceId, &d_a, &d_b, &d_c, M, N, K, supportsManagedMemory != 0);

#ifdef VERIFY
    cVerify = new float[M * N];
    matrixMultiplyCPU(d_a, d_b, cVerify, M, N, K, supportsManagedMemory != 0);
#endif // VERIFY

    std::cout << "grid_size,block_size,elapsed_msec" << std::endl;

    for (int numThreads = 16; numThreads <= 2500 && result == 0; numThreads += 32)
    {
        result = run(deviceId, d_a, d_b, d_c, cVerify, M, N, K, numThreads, strategy, supportsManagedMemory != 0);
    }

    cleanup(d_a, d_b, d_c);

    if (cVerify != nullptr)
    {
        delete[] cVerify;
    }

    return result;
}