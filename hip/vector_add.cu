#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

using float16_t = _Float16;

__global__ void vectorAdd(float16_t *a, float16_t *b, float16_t *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

void run()
{
    constexpr int N = 2 << 16;
    constexpr int blockSize = 1024;
    constexpr float16_t tolerance = 1e-5;

    const size_t bytes = N * sizeof(double);

    // allocate host memory
    float16_t *CPUArrayA = reinterpret_cast<float16_t *>(malloc(bytes));
    float16_t *CPUArrayB = reinterpret_cast<float16_t *>(malloc(bytes));
    float16_t *CPUArrayC = reinterpret_cast<float16_t *>(malloc(bytes));

    // initialize host memory
    for (int i = 0; i < N; ++i)
    {
        CPUArrayA[i] = static_cast<float16_t>(i);
        CPUArrayB[i] = static_cast<float16_t>(i);
    }

    // allocate device memory
    float16_t *GPUArrayA;
    float16_t *GPUArrayB;
    float16_t *GPUArrayC;

    HIP_ASSERT(hipMalloc(&GPUArrayA, bytes));
    HIP_ASSERT(hipMalloc(&GPUArrayB, bytes));
    HIP_ASSERT(hipMalloc(&GPUArrayC, bytes));

    // initialize device memory
    HIP_ASSERT(hipMemcpy(GPUArrayA, CPUArrayA, bytes, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(GPUArrayB, CPUArrayB, bytes, hipMemcpyHostToDevice));

    const auto gridSize = static_cast<int>(ceil((float)N / blockSize));

    vectorAdd<<<gridSize, blockSize>>>(GPUArrayA, GPUArrayB, GPUArrayC, N);
    hipDeviceSynchronize();

    HIP_ASSERT(hipMemcpy(CPUArrayC, GPUArrayC, bytes, hipMemcpyDeviceToHost));

    // verify
    bool failed = false;
    for (int i = 0; i < N; ++i)
    {
        float16_t verify = CPUArrayA[i] + CPUArrayB[i];
        if (abs(static_cast<float>(verify - CPUArrayC[i])) > tolerance)
        {
            std::cout << "Error at [" << i << "], expected: " << static_cast<float>(verify)
                      << ", got: " << static_cast<float>(CPUArrayC[i]) << std::endl;
            failed = true;
        }
    }

    if (!failed) {
        std::cout << "Success" << std::endl;
    }

    // release device memory
    HIP_ASSERT(hipFree(GPUArrayA));
    HIP_ASSERT(hipFree(GPUArrayB));
    HIP_ASSERT(hipFree(GPUArrayC));

    // release host memory
    free(CPUArrayA);
    free(CPUArrayB);
    free(CPUArrayC);
}

int main(int argc, const char **argv)
{
    run();
    return 0;
}