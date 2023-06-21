#include <hip/hip_runtime.h>
#include <fstream>
#include <iostream>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

__global__ void imageGamma(uint8_t *data, float gamma, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        data[i] = pow(data[i] / 255.0, gamma) * 255.0;
    }
}

uint8_t *loadImage(const char *filename, int *size)
{
    *size = 0;

    std::ifstream file;
    file.open(filename, std::ios::in | std::ios::binary | std::ios::ate);
    const int fileSize = file.tellg(); // std::ios::ate does a SEEK_END

    if (fileSize < 0)
    {
        throw std::runtime_error("Unexpected file size");
    }

    uint8_t *data = new uint8_t[fileSize];
    file.seekg(file.beg); // SEEK_SET to 0
    file.read(reinterpret_cast<char *>(data), fileSize);

    *size = fileSize;
    return data;
}

void saveImage(const uint8_t *data, int size, const char *filename)
{
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::binary);
    file.write(reinterpret_cast<const char *>(data), size);
}

void run(const char *inFile, const char *outFile)
{
    constexpr int blockSize = 256;
    constexpr float gamma = 4.0;

    // load image from file
    int n;
    uint8_t *CPUdata = loadImage(inFile, &n);
    const int gridSize = (n + blockSize - 1) / blockSize;

    // initialize device memory
    uint8_t *GPUdata;
    HIP_ASSERT(hipMalloc(&GPUdata, n));
    HIP_ASSERT(hipMemcpy(GPUdata, CPUdata, n * sizeof(uint8_t), hipMemcpyHostToDevice));

    imageGamma<<<gridSize, blockSize>>>(CPUdata, gamma, n);

    HIP_ASSERT(hipMemcpy(CPUdata, GPUdata, n * sizeof(uint8_t), hipMemcpyDeviceToHost));

    // save image to result file
    saveImage(CPUdata, n, outFile);

    HIP_ASSERT(hipFree(GPUdata));
    delete[] CPUdata;
}

int main(int argc, const char **argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " input_file output_file" << std::endl;
        return -1;
    }
    run(argv[1], argv[2]);
    return 0;
}