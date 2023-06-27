#pragma once

#include <hip/hip_runtime.h>

void HIP_ASSERT(hipError_t e)
{
    if (e != hipSuccess)
    {
        std::cout << hipGetErrorString(e) << std::endl;
        assert(false);
    }
}
