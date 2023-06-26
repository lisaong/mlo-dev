#include <hip/hip_runtime.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

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

__global__ void reduceSum(float16_t *input, float_t *output, int n)
{
    extern __shared__ float localSum[];
    const int gridSize = blockDim.x * gridDim.x;

    // parallel local sum
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (; i < n; i += gridSize)
    {
        sum += input[i];
    }
    localSum[threadIdx.x] = sum;
    __syncthreads();

    // reduction:
    //  s = 128
    //  localSum[0] += localSum[128]
    //  localSum[1] += localSum[129]
    //  ...
    //  localSum[127] += localSum[255]
    //  __syncthreads()
    //
    //  s = 64
    //  localSum[0] += localSum[64]
    //  localSum[1] += localSum[65]
    //  ...
    //  localSum[63] += localSum[127]
    //  __syncthreads()
    //
    //  s = 32
    //  localSum[0] += localSum[32]
    //  localSum[1] += localSum[33]
    //  ...
    //  localSum[31] += localSum[63]
    //  __syncthreads()
    //  etc

    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
        {
            localSum[threadIdx.x] += localSum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // write the per-block sum
    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = localSum[0];
    }
}

int run(int deviceId, int numBlocks)
{
    constexpr int numThreads = 256;
    constexpr int N = 10485760;
    constexpr int sharedMemorySize = numThreads * sizeof(float);

    float16_t *d_a;
    float *d_b;
    HIP_ASSERT(hipMallocManaged(&d_a, N * sizeof(float16_t)));
    HIP_ASSERT(hipMallocManaged(&d_b, numBlocks * sizeof(float)));

    HIP_ASSERT(hipMemPrefetchAsync(d_a, N * sizeof(float16_t), deviceId));
    HIP_ASSERT(hipMemPrefetchAsync(d_b, numBlocks * sizeof(float), deviceId));

    init<<<numBlocks, numThreads>>>(d_a, N);

    {
        std::stringstream ss;
        ss << numBlocks << "," << numThreads;

        TimedRegion r(ss.str());
        reduceSum<<<numBlocks, numThreads, sharedMemorySize, 0>>>(d_a, d_b, N);
        hipDeviceSynchronize();
    }

    // Finalize the sum
    float sum = 0.0f;
    for (int i = 0; i < numBlocks; ++i)
    {
        sum += d_b[i];
    }

    // Verify
    float expectedSum = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        expectedSum += d_a[i];
    }

    if (abs(sum - expectedSum) > 1e-5)
    {
        std::cout << "Error: sum " << sum << ", expected sum " << expectedSum << std::endl;
        return -1;
    }

    HIP_ASSERT(hipFree(d_a));
    HIP_ASSERT(hipFree(d_b));

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
    for (int numBlocks = 32; numBlocks <= 2048 && result == 0; numBlocks += 32)
    {
        result = run(deviceId, numBlocks);
    }
    return result;
}