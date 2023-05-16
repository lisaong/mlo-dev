#pragma once

#include <cstdint>

#if DEBUG
constexpr uint32_t N = 16;
#else
constexpr uint32_t N = 1024;
#endif // DEBUG

constexpr uint32_t kBandDim = N;
constexpr uint32_t kBlockDim = 1;
constexpr uint32_t kBlockDimStep = 1;
constexpr uint32_t kMaxBlockDim = 40;
constexpr uint32_t kNumberOfOps = 2 * N * N * N;
constexpr uint32_t kMillisecondsInSeconds = 1000;
constexpr uint32_t kTimelimit = 10 * kMillisecondsInSeconds;