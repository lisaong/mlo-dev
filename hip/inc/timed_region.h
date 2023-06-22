#pragma once
#include <iostream>

#include <hip/hip_runtime.h>

// cf. https://github.com/ROCm-Developer-Tools/hipamd/blob/develop/src/hip_event.cpp

#ifndef HIP_ASSERT
#define HIP_ASSERT(x) (assert((x) == hipSuccess))
#endif

struct TimedRegion
{
    hipEvent_t start;
    hipEvent_t end;

    TimedRegion()
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
        std::cout << " >> Elapsed time (ms): " << millisecs << std::endl;

        HIP_ASSERT(hipEventDestroy(start));
        HIP_ASSERT(hipEventDestroy(end));
    }
};