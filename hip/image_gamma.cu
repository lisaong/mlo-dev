#include <hip/hip_runtime.h>
#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "inc/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "inc/stb_image_write.h"

// cf. https://gitlab.com/syifan/hipbookexample/-/blob/main/Chapter5/ImageGamma/main.cpp

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

__global__ void imageGamma(uint8_t *data, float gamma, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        data[i] = pow(data[i] / 255.0, gamma) * 255.0;
    }
}

int run(const char *inFile, const char *outFile)
{
    constexpr int blockSize = 256;
    constexpr float gamma = 4.0;

    // load image from file
    int width, height, channels;
    uint8_t* CPUdata = stbi_load(inFile, &width, &height, &channels, /*reg_comp*/0);
    if (CPUdata == nullptr) {
        std::cout << "Failed to load image from " << inFile << std::endl;
        return -1;
    }

    int n = width * height * channels;
    const int gridSize = (n + blockSize - 1) / blockSize;

    // initialize device memory
    uint8_t *GPUdata;
    HIP_ASSERT(hipMalloc(&GPUdata, n));
    HIP_ASSERT(hipMemcpy(GPUdata, CPUdata, n * sizeof(uint8_t), hipMemcpyHostToDevice));

    imageGamma<<<gridSize, blockSize>>>(CPUdata, gamma, n);
    hipDeviceSynchronize();

    HIP_ASSERT(hipMemcpy(CPUdata, GPUdata, n * sizeof(uint8_t), hipMemcpyDeviceToHost));

    // save image to result file
    stbi_write_jpg(outFile, width, height, channels, CPUdata, /*quality*/100);

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
    return run(argv[1], argv[2]);
}