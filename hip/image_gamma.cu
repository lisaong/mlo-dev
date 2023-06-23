#include <hip/hip_runtime.h>
#include <iostream>
#include <sstream>

#include "inc/image.h"
#include "inc/timed_region.h"

// cf. https://gitlab.com/syifan/hipbookexample/-/blob/main/Chapter5/ImageGamma/main.cpp

#ifndef HIP_ASSERT
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif

__global__ void imageGamma(uint8_t *data, float gamma, int n)
{
    const int stride = blockDim.x * gridDim.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (; i < n; i += stride)
    {
        float value = data[i] / 255.0f;
        value = pow(value, gamma);
        data[i] = static_cast<uint8_t>(value * 255.0f);
    }
}

int run(const char *inFile, const char *outFile, int gridSize)
{
    constexpr float gamma = 4.0;
    constexpr int blockSize = 256;

    // load image from file
    int width, height, channels;
    uint8_t *CPUdata = stbi_load(inFile, &width, &height, &channels, /*reg_comp*/ 0);
    if (CPUdata == nullptr)
    {
        std::cout << "Failed to load image from " << inFile << std::endl;
        return -1;
    }

    const int n = width * height * channels;

    // initialize device memory
    uint8_t *GPUdata;
    HIP_ASSERT(hipMalloc(&GPUdata, n));
    HIP_ASSERT(hipMemcpy(GPUdata, CPUdata, n * sizeof(uint8_t), hipMemcpyHostToDevice));

    {
        std::stringstream ss;
        ss << gridSize << "," << blockSize;
        TimedRegion r(ss.str());

        imageGamma<<<gridSize, blockSize>>>(GPUdata, gamma, n);
        hipDeviceSynchronize();
    }

    HIP_ASSERT(hipMemcpy(CPUdata, GPUdata, n * sizeof(uint8_t), hipMemcpyDeviceToHost));

    // save image to result file
    stbi_write_jpg(outFile, width, height, channels, CPUdata, /*quality*/ 100);

    HIP_ASSERT(hipFree(GPUdata));
    stbi_image_free(CPUdata);
    return 0;
}

int main(int argc, const char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " input_file output_file" << std::endl;
        return -1;
    }

    // try different block sizes
    std::cout << "grid_size,block_size,elapsed_msec" << std::endl;
    int result = 0;
    for (int gridSize = 32; gridSize <= 2048 && result == 0; gridSize += 32)
    {
        result = run(argv[1], argv[2], gridSize);
    }
    return result;
}