#include <hip/hip_runtime.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "inc/image.h"
#include "inc/timed_region.h"


#ifndef HIP_ASSERT
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif


__global__ void conv2d(uint8_t *image, float *mask, int width, int height, int maskWidth, int maskHeight)
{

}


int run(const char *inFile, const char *outFile, uint32_t blockSize)
{
    // load image from file
    int width, height, channels;
    uint8_t *image = stbi_load(inFile, &width, &height, &channels, /*reg_comp*/ 0);
    if (image == nullptr)
    {
        std::cout << "Failed to load image from " << inFile << std::endl;
        return -1;
    }

    constexpr int maskWidth = 200;
    constexpr int maskHeight = 200;
    std::vector<float> mask(maskWidth * maskHeight * channels);
    for (int i = 0; i < mask.size(); ++i)
    {
        mask[i] = 1.0f / maskWidth / maskHeight / channels;
    }

    // allocate and initialize GPU memory
    uint8_t *d_image;
    float *d_mask;
    HIP_ASSERT(hipMalloc(&d_image, width * height * channels * sizeof(uint8_t)));
    HIP_ASSERT(hipMalloc(&d_mask, mask.size() * sizeof(float)));
    HIP_ASSERT(hipMemcpy(d_image, image, width * height * channels * sizeof(uint8_t), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_mask, mask.data(), mask.size() * sizeof(float), hipMemcpyHostToDevice));

    // calculate grid size and launch kernel
    dim3 blockSizes = {blockSize, blockSize, 1};
    dim3 gridSizes = {(width + blockSizes.x - 1) / blockSizes.x, (height + blockSizes.y - 1) / blockSizes.y, 1};

    {
        std::stringstream ss;
        ss << gridSizes.x << "," << blockSizes.x;
        TimedRegion r(ss.str());

        conv2d<<<gridSizes, blockSizes>>>(d_image, d_mask, width, height, maskWidth, maskHeight);
        HIP_ASSERT(hipDeviceSynchronize());
    }

    HIP_ASSERT(hipMemcpy(image, d_image, width * height * channels * sizeof(uint8_t), hipMemcpyDeviceToHost));

    // save image to result file
    stbi_write_jpg(outFile, width, height, channels, image, /*quality*/ 100);

    HIP_ASSERT(hipFree(d_image));
    HIP_ASSERT(hipFree(d_mask));

    stbi_image_free(image);
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
    for (uint32_t blockSize = 32; blockSize <= 1024 && result == 0; blockSize += 32)
    {
        result = run(argv[1], argv[2], blockSize);
    }
    return 0;
}