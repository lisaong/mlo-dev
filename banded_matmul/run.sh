#!/bin/sh

clang-format -i *.cu *.h

echo "Running bmm: naive"
rm -rf bmm
nvcc --use_fast_math --std=c++17 -o bmm bmm.cu
./bmm 0
echo "=================="

rm -rf bmm_smem
nvcc --use_fast_math --std=c++17 -o bmm_smem bmm_smem.cu

echo "Running bmm_smem: T1 in shared memory"
./bmm_smem 0 0
echo "=================="

echo "Running bmm_smem: T0 and T1 in shared memory"
./bmm_smem 0 1
echo "=================="

echo "Running bmm_smem: T0 and T1 in shared memory, T2 column major"
./bmm_smem 0 2
echo "=================="

# rm -rf bmm_wmma
# nvcc --gpu-architecture=compute_86 --use_fast_math --std=c++17 -o bmm_wmma bmm_wmma.cu
# ./bmm_wmma 0