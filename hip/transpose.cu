#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>
#include <vector>

#include "inc/assert.h"
#include "inc/timed_region.h"

using float16_t = _Float16;

#define CDIV(n, block) (n + block - 1) / block

// cf. https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

template <typename T>
__global__ void transposeNaive(T *A, T *B, uint64_t M)
{
    // naive transposition without memory coalescing
    // only square matrices are handled here
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    //  (N, M)    (M, N)
    // B[j, i] = A[i, j]
    if (i < M && j < M)
    {
        B[j * M + i] = A[i * M + j];
    }
}

template <typename T>
__global__ void transposeCoalesced(T *A, T *B, uint64_t M)
{
    extern __shared__ T subTileA[]; // blockDim.y * blockDim.x

    // copy A to subtileA, accesses along the j dimension will be coalesced
    const int iTileSize = blockDim.y;
    const int jTileSize = blockDim.x;
    const int iTile = blockIdx.y * iTileSize;
    const int jTile = blockIdx.x * jTileSize;

    // shared memory offsets
    const int ii = threadIdx.y;
    const int jj = threadIdx.x;

    // subtileA[ii, jj] = A[i, j]
    subTileA[ii * jTileSize + jj] = A[(iTile + ii) * M + (jTile + jj)];
    __syncthreads();

    // copy a column of subTileA to a row of B, so that accesses
    // along the j dimension will be coalesced
    B[(iTile + ii) * M + (jTile + jj)] = subTileA[jj * iTileSize + ii];
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
    constexpr int M = 32768;
    constexpr int tileDim = 16;
    std::vector<float16_t> A;
    std::vector<float16_t> B;

    A.resize(M, M);
    B.resize(M, M);

    for (int i = 0; i < A.size(); ++i)
    {
        A[i] = static_cast<float16_t>(rand()) / static_cast<float16_t>(RAND_MAX);
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

    transposeCoalesced<<<dim3(gridDim, gridDim), dim3(tileDim, tileDim)>>>(d_A, d_B, M);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());

    HIP_ASSERT(hipMemcpy(B.data(), d_B, B.size() * sizeof(float16_t), hipMemcpyDeviceToHost));
    verify(A.data(), B.data(), M);

    HIP_ASSERT(hipFree(d_A));
    HIP_ASSERT(hipFree(d_B));

    return 0;
}