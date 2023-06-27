#pragma once
#include <iostream>

#include <hip/hip_runtime.h>

#include "assert.h"

// cf. https://github.com/ROCm-Developer-Tools/hipamd/blob/develop/src/hip_event.cpp

struct TimedRegion
{
    hipEvent_t start;
    hipEvent_t end;
    std::string prefix;

    TimedRegion(std::string prefix = "") : prefix(prefix)
    {
        HIP_ASSERT(hipEventCreate(&start));
        HIP_ASSERT(hipEventCreate(&end));
        HIP_ASSERT(hipEventRecord(start, 0));
    }

    ~TimedRegion()
    {
        HIP_ASSERT(hipEventRecord(end, 0));
        HIP_ASSERT(hipEventSynchronize(end)); // wait for event to complete

        float millisecs = 0;
        HIP_ASSERT(hipEventElapsedTime(&millisecs, start, end));
        std::cout << prefix << "," << millisecs << std::endl;

        HIP_ASSERT(hipEventDestroy(start));
        HIP_ASSERT(hipEventDestroy(end));
    }
};