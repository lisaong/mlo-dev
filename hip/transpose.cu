#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "inc/assert.h"
#include "inc/timed_region.h"

using float16_t = _Float16;

#define CDIV(n, block) (n + block - 1) / block

// cf. https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// CUDA uses a slightly different way to avoid bank conflicts

template <typename T>
__global__ void transposeNaive(const T *A, T *B, uint64_t M)
{
    // naive transposition without memory coalescing
    // only square matrices are handled here
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    //  (N, M)    (M, N)
    // B[j, i] = A[i, j]
    if (i < M && j < M)
    {
        // j is the minor axis
        // accessing B column-wise, accessing A row-wise
        B[j * M + i] = A[i * M + j];
    }
}

template <typename T>
__global__ void transposeCoalesced(const T *A, T *B, uint64_t M)
{
    extern __shared__ T tile[]; // blockDim.y * blockDim.x

    const int iTileSize = blockDim.y;
    const int jTileSize = blockDim.x;
    const int iTile = blockIdx.y * iTileSize;
    const int jTile = blockIdx.x * jTileSize;

    // shared memory offsets
    const int ii = threadIdx.y;
    const int jj = threadIdx.x;

    // copy a row of A to a row of tile
    tile[ii * jTileSize + jj] = A[(iTile + ii) * M + (jTile + jj)];
    __syncthreads();

    // copy a column of tile to a row of B
    // To do this, we keep jj as the minor axis when traversing B
    // so that memory reads (threadIdx.x) will be close by
    //
    // x ->
    //      thread(0, 0) (1, 0), (2, 0)
    // y
    // |
    // v
    //
    // row of B (jj is the minor axis)   column of A (ii is the minor axis)
    //
    // (Note that this won't work for rectangular matrices)
    B[(jTile + ii) * M + (iTile + jj)] = tile[jj * iTileSize + ii];
}

template <typename T>
bool verify(T *A, T *B, uint64_t M)
{
    bool match = true;
    for (int i = 0; i < M && match; ++i)
    {
        for (int j = 0; j < M && match; ++j)
        {
            if (B[i * M + j] != A[j * M + i])
            {
                std::cout << "[" << i << ", " << j << "]: expected "
                          << static_cast<float>(A[j * M + i])
                          << ", got "
                          << static_cast<float>(B[i * M + j])
                          << std::endl;
                match = false;
            }
        }
    }
    return match;
}

int main(int argc, const char **argv)
{
    constexpr int M = 1024; // 32768;
    constexpr int tileDim = 16;
    std::vector<float16_t> A;
    std::vector<float16_t> B;

    A.resize(M * M);
    B.resize(M * M);

    for (int i = 0; i < A.size(); ++i)
    {
        A[i] = static_cast<float16_t>(i) / (M / 4);
    }

    float16_t *d_A;
    float16_t *d_B;

    HIP_ASSERT(hipMalloc(&d_A, A.size() * sizeof(float16_t)));
    HIP_ASSERT(hipMalloc(&d_B, B.size() * sizeof(float16_t)));

    HIP_ASSERT(hipMemcpy(d_A, A.data(), A.size() * sizeof(float16_t), hipMemcpyHostToDevice));

    constexpr int gridDim = CDIV(M, tileDim);

    transposeNaive<<<dim3(gridDim, gridDim), dim3(tileDim, tileDim)>>>(d_A, d_B, M);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());

    HIP_ASSERT(hipMemcpy(B.data(), d_B, B.size() * sizeof(float16_t), hipMemcpyDeviceToHost));
    verify(A.data(), B.data(), M);

    constexpr int sharedMemorySize = tileDim * tileDim * sizeof(float16_t);
    transposeCoalesced<<<dim3(gridDim, gridDim), dim3(tileDim, tileDim), sharedMemorySize>>>(d_A, d_B, M);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());

    HIP_ASSERT(hipMemcpy(B.data(), d_B, B.size() * sizeof(float16_t), hipMemcpyDeviceToHost));
    verify(A.data(), B.data(), M);

    HIP_ASSERT(hipFree(d_A));
    HIP_ASSERT(hipFree(d_B));

    return 0;
}