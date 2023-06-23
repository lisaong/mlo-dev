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

__global__ void sum(float16_t *input, float_t *output, int n)
{
}

int run(int numBlocks)
{
    constexpr int numThreads = 256;
    constexpr int N = 10485760;

    std::vector<float16_t> a(N);
    std::vector<float> b(numBlocks);
    std::fill(a.begin(), a.end(), static_cast<float>(rand() - RAND_MAX / 2) / static_cast<float>(RAND_MAX));

    float16_t *d_a;
    float *d_b;
    HIP_ASSERT(hipMalloc(&d_a, a.size() * sizeof(float16_t)));
    HIP_ASSERT(hipMalloc(&d_b, b.size() * sizeof(float)));
    HIP_ASSERT(hipMemcpy(d_a, a.data(), a.size() * sizeof(float16_t), hipMemcpyHostToDevice));

    {
        std::stringstream ss;
        ss << numBlocks << "," << numThreads;

        TimedRegion r(ss.str());
        sum<<<numBlocks, numThreads>>>(d_a, d_b, N);
        hipDeviceSynchronize();
    }

    HIP_ASSERT(hipMemcpy(b.data(), d_b, b.size() * sizeof(float), hipMemcpyDeviceToHost));

    // Finalize the sum
    float sum = 0.0f;
    for (int i = 0; i < numBlocks; ++i)
    {
        sum += b[i];
    }

    // Verify
    float expectedSum = 0.0f;
    for (int i = 0; i < a.size(); ++i)
    {
        expectedSum += a[i];
    }

    if (abs(sum - expectedSum) > 1e-5)
    {
        std::cout << "Error: sum " << sum << ", expected sum " << expectedSum << std::endl;
        return -1;
    }

    return 0;
}

int main(int argc, const char **argv)
{
    std::cout << "grid_size,block_size,elapsed_msec" << std::endl;
    int result = 0;
    for (int numBlocks = 32; numBlocks <= 700 && result == 0; numBlocks += 32)
    {
        result = run(numBlocks);
    }
    return result;
}