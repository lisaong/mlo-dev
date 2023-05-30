#pragma once

#include <cstdint>

// constexpr uint32_t N = 1024;
constexpr uint32_t N = 64;

constexpr uint32_t kBandDim = N;
constexpr uint32_t kBlockDimX = 64;
constexpr uint32_t kBlockDimXStep = 32;
constexpr uint32_t kBlockDimXMax = 256;
constexpr uint32_t kMaxBlockDim = 1024;
constexpr uint32_t kNumberOfOps = 2 * N * N * N;
constexpr uint32_t kMillisecondsInSeconds = 1000;
constexpr uint32_t kTimelimit = 10 * kMillisecondsInSeconds;
constexpr float kEpsilon = 5e-2f;
