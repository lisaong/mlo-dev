#!/bin/sh
GPU_ID=0
clang-format -i *.cu *.h

# echo "Running bmm: naive"
# rm -rf bmm
# nvcc --use_fast_math --std=c++17 -o bmm bmm.cu
# ./bmm $GPU_ID
# echo "=================="

# rm -rf bmm_smem
# nvcc --use_fast_math --std=c++17 -o bmm_smem bmm_smem.cu

# echo "Running bmm_smem: T1 in shared memory"
# ./bmm_smem $GPU_ID 0
# echo "=================="

# echo "Running bmm_smem: T0 and T1 in shared memory"
# ./bmm_smem $GPU_ID 1
# echo "=================="

# echo "Running bmm_smem: T0 and T1 in shared memory, T2 column major"
# ./bmm_smem $GPU_ID 2
# echo "=================="

rm -rf bmm_dbuf

NVCC_FLAGS="-use_fast_math --std=c++20 --gpu-architecture=compute_86 -diag-suppress 20054"
NVCC_DEBUG_FLAGS="$NVCC_FLAGS --maxrregcount=64 -g -G"

# nvcc $NVCC_DEBUG_FLAGS -o bmm_dbuf bmm_dbuf.cu
# cuda-gdb bmm_dbuf

nvcc $NVCC_FLAGS -o bmm_dbuf bmm_dbuf.cu

# echo "Running bmm_dbuf: cooperative group synchronous copy"
# ./bmm_dbuf $GPU_ID 0
# echo "=================="

# echo "Running bmm_dbuf: cooperative group asynchronous copy"
# ./bmm_dbuf $GPU_ID 1
# echo "=================="

echo "Running bmm_dbuf: cooperative group asynchronous copy with arrive-wait barriers"
./bmm_dbuf $GPU_ID 2
echo "=================="

# rm -rf bmm_wmma
# nvcc --gpu-architecture=compute_86 --use_fast_math --std=c++17 -o bmm_wmma bmm_wmma.cu
# ./bmm_wmma $GPU_ID