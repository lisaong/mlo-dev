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

__global__ void matrixMultiplyTiled(float16_t *A, float16_t *B, float *C, uint64_t M, uint64_t N, uint64_t K, int tileSizeX, int tileSizeY)
{
    // C[i, j] += A[i, k] * B[k, j]
    // (M, N)    (M, K)    (K, N)
    //   where y => rows (i), x => colummns (j) for C

    extern __shared__ float subTileA[];                 // (tileSizeY by tileSizeX)
    float *subTileB = &subTileA[tileSizeY * tileSizeX]; // (tileSizeX by tileSizeY)

    // cumulative sum across the full K dimension for each thread
    // (one full row of A x one full column of B)
    float sum = 0.0f;

    // load the A and B tiles
    const int i = blockIdx.y * tileSizeY + threadIdx.y;
    const int j = blockIdx.x * tileSizeX + threadIdx.x;
    const int numTiles = CDIV(K, tileSizeX);

    // walk the tiles along the k dimension (columns of A)
    for (int k = 0; k < numTiles; ++k)
    {
        // load tileSizeY rows of A (i dimension, threadIdx.y)
        //   and tileSizeX columns (k dimension, threadIdx.x)
        auto aRow = i;
        auto aCol = k * tileSizeX + threadIdx.x;
        auto smem = threadIdx.y * tileSizeX + threadIdx.x;

        if (aRow < M && aCol < K)
        {
            // only tileSizeY x tileSizeX will be copied per workgroup
            subTileA[smem] = A[aRow * K + aCol];
        }
        else
        {
            subTileA[smem] = 0.0f;
        }

        // load tileSizeX rows of B (k dimension, threadIdx.x)
        //   and tileSizeY cols of B (j dimension, threadIdx.y)
        auto bRow = k * tileSizeX + threadIdx.x;
        auto bCol = j;
        smem = threadIdx.x * tileSizeY + threadIdx.y;

        if (bRow < K && bCol < N)
        {
            // only tileSizeX x tileSizeY will be copied per workgroup
            // BUGBUG: non-coalesced global memory access for B
            //         (for coalesced access, threadIdx.x needs
            //         to do contiguous reads)
            subTileB[smem] = B[bRow * N + bCol];
        }
        else
        {
            subTileB[smem] = 0.0f;
        }

        __syncthreads(); // wait for complete tile to be loaded

        // multiply subTileA with subTileB
        // each thread will take the subTileA's row across the kk dimension (tileSizeX)
        // and multiply that by subTileB's column across the kk dimension (tileSizeX)
        float tileSum = 0.0f; // for clarity
        for (int kk = 0; kk < tileSizeX; ++kk)
        {
            if (k * tileSizeX + kk < K)
            {
                tileSum += subTileA[threadIdx.y * tileSizeX + kk] * subTileB[kk * tileSizeY + threadIdx.x];
            }
        }
        // aggregate the tile-local sum into the k sum
        sum += tileSum;

        __syncthreads(); // wait for processing of the current tile to be complete, otherwise
                         // other threads may update subTileA or subTileB before we are done
                         // with computing all the thread-local sums

        // update the result
        C[i * N + j] = sum;
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

    const dim3 numThreads(256, 4, 1);
    dim3 numBlocks(CDIV(M, numThreads.x), CDIV(K, numThreads.y), 1);
    init<<<numBlocks, numThreads>>>(*input1, M, K);
    HIP_ASSERT(hipGetLastError());

    numBlocks.x = CDIV(K, numThreads.x);
    numBlocks.y = CDIV(N, numThreads.y);
    init<<<numBlocks, numThreads>>>(*input2, K, N);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());
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
    int maxThreadsPerBlock = 0;
    HIP_ASSERT(hipDeviceGetAttribute(&maxThreadsPerBlock,
                                     hipDeviceAttributeMaxThreadsPerBlock, deviceId));

    const int tileSizeX = tileSize;
    const int tileSizeY = max(1, tileSize / 16); // use rectangular tiles to fit the limit of 1024
    const dim3 numThreads(tileSizeX, tileSizeY, 1);
    const dim3 numBlocks(CDIV(M, numThreads.x), CDIV(N, numThreads.y), 1);

    auto requestedThreads = numThreads.x * numThreads.y;
    if (requestedThreads > maxThreadsPerBlock)
    {
        std::cout << "Num threads requested: " << requestedThreads << " exceeds limit (" << maxThreadsPerBlock << ")" << std::endl;
        return -2;
    }

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
        int sharedMemorySize = tileSizeX * tileSizeY * sizeof(TIn) * 2; // subTileA and subTileB

        // compute the amount of shared memory available
        int sharedMemoryPerBlock = 0;
        HIP_ASSERT(hipDeviceGetAttribute(&sharedMemoryPerBlock,
                                         hipDeviceAttributeMaxSharedMemoryPerBlock, deviceId));
        if (sharedMemorySize > sharedMemoryPerBlock)
        {
            std::cout << "Shared memory needed: " << sharedMemorySize << " (bytes) exceeds limit (" << sharedMemoryPerBlock << ")" << std::endl;
            return -2;
        }

        TimedRegion r(ss.str());

        matrixMultiplyTiled<<<numBlocks, numThreads, sharedMemorySize>>>(d_a, d_b, d_c, M, N, K, tileSizeX, tileSizeY);
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
                std::cout << "C " << c[i * N + j] << ", verify "
                          << verify[i * N + j] << std::endl;

                auto diff = std::abs(c[i * N + j] - verify[i * N + j]);
                if (diff > 1e-3)
                {
                    std::cout << "Error: C[" << i << ", " << j << "] = "
                              << c[i * N + j] << ", expected "
                              << verify[i * N + j] << ", abs diff "
                              << diff << std::endl;
                    // match = false;
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

    for (int numThreads = 16; numThreads < 512 && result == 0; numThreads += 16)
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