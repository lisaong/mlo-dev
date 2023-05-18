// Banded matrix multiplication using WMMA
// cf: simple_wmma_tf32gemm from
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/3_CUDA_Features/tf32TensorCoreGemm/tf32TensorCoreGemm.cu

#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

#include <cstdint>

// #define DEBUG 1
#include "constants.h"
#include "utils.h"

// constexpr int WARP_SIZE = 32;
// constexpr int WARPS_PER_BLOCK = 8;
// constexpr int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// constexpr int M_TILES = N / WMMA_M;
// constexpr int N_TILES = N / WMMA_N;
// constexpr int  K_TILES = N / WMMA_K;

// TODO: Not yet correct
__global__ void bandedMatMul_wmma(int n0, int n1, int n2, float *t0,
                                  const half *t1, const half *t2) {

  // Leading dimensions
  int lda = n2;
  int ldb = n2;
  int ldc = n1;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < n2; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * N;
    int bRow = i;

    // Bounds checking
    if (aRow < n0 && aCol < n2 && bRow < n2 && bCol < n1) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, t1 + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, t2 + (aRow + bRow) + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of C and add our result
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < n0 && cCol < n1) {
    wmma::load_matrix_sync(c_frag, t0 + cCol + cRow * ldc, ldc,
                           wmma::mem_row_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] += acc_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(t0 + cCol + cRow * ldc, c_frag, ldc,
                            wmma::mem_row_major);
  }
}

void run(int deviceId) {

  const int n0 = N; // n0: number of rows in T0 and T1
  const int n1 = N; // n1: number of columns in T0 and T2
  const int n2 = N; // n2: inner or shared dimension, i.e.
                    //     number of columns in T1 and number of rows in T2

  Matrix<float> T0(n0, n1);            // output
  BandedMatrix<half> T1(n0, kBandDim); // input
  Matrix<half> T2(T1.columns(), n1);   // input

  CHECK(cudaMallocManaged(&T0.data, T0.size()));
  CHECK(cudaMallocManaged(&T1.data, T1.size()));
  CHECK(cudaMallocManaged(&T2.data, T2.size()));

  // Initialize
  dim3 threads(kBlockDimX, kMaxBlockDim / kBlockDimX, 1);
  dim3 blocks(n0 / threads.x, n1 / threads.y, 1);

  T0.randomInit(123);
  CHECK(cudaMemPrefetchAsync(T0.data, T0.size(), deviceId));
  initBandedWith<<<blocks, threads>>>(static_cast<half>(22.0f), T1.data,
                                      T1.rows(), T1.columns(), T1.band());
  initWith<<<blocks, threads>>>(static_cast<half>(33.0f), T2.data,
                                      T2.rows(), T2.columns());
  CHECK(cudaDeviceSynchronize());

  // Verify
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x =
      (n0 + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (n1 + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

  bandedMatMul_wmma<<<gridDim, blockDim>>>(n0, n1, n2, T0.data, T1.data,
                                           T2.data);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  if (checkCorrectness(n0, n1, n2, T0, T1, T2)) {

#if 0
    // Benchmark
    cudaEvent_t _start;
    cudaEvent_t _stop;
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);

    std::cout << "GridDim,BlockDim,FLOPS,GFLOPS" << std::endl;

    // TODO: fix
    // Try different block sizes
    for (uint32_t blockDim = WARP_SIZE; blockDim <= kMaxBlockDim;
         blockDim += WARP_SIZE) {

      threads.x = blockDim;
      threads.y = kMaxBlockDim / blockDim;
      blocks.x = ceildiv(n0, threads.x);
      blocks.y = ceildiv(n1, threads.y);

      try {
        double elapsedTimeMilliseconds = 0.0f;
        uint64_t iterations = 0;
        float duration = 0.0f;

        // Runs the function until 10 seconds has elapsed
        cudaEventRecord(_start);
        while (elapsedTimeMilliseconds < kTimelimit) {
          bandedMatMul_wmma<<<blocks, threads>>>(n0, n1, n2, T0.data, T1.data,
                                                 T2.data);

          CHECK(cudaGetLastError());
          CHECK(cudaDeviceSynchronize());

          cudaEventRecord(_stop);
          cudaEventSynchronize(_stop);
          cudaEventElapsedTime(&duration, _start, _stop);
          elapsedTimeMilliseconds += duration;
          iterations++;
        }

        const double flops = iterations * kNumberOfOps /
                             (elapsedTimeMilliseconds / kMillisecondsInSeconds);
        std::cout << blocks.x << "," << threads.x << "," << flops << ","
                  << flops / 1e9 << std::endl;
      } catch (const std::exception &e) {
        std::cout << "Skipping Blocksize: " << blockDim << ", " << e.what()
                  << std::endl;
        continue;
      }
    }

    cudaEventDestroy(_start);
    cudaEventDestroy(_stop);
#endif
  }

  cudaFree(T0.data);
  cudaFree(T1.data);
  cudaFree(T2.data);
}

int main(int argc, const char **argv) {
  int deviceId;

  if (argc > 1) {
    deviceId = atoi(argv[1]);
    CHECK(cudaSetDevice(deviceId));
  } else {
    CHECK(cudaGetDevice(&deviceId));
  }
  std::cout << "Using device " << deviceId << std::endl;

  run(deviceId);
  return 0;
}