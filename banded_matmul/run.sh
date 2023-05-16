#!/bin/sh

clang-format -i *.cu *.h

rm -rf bmm
nvcc --use_fast_math --std=c++17 -o bmm bmm.cu
./bmm 0

rm -rf bmm_smem
nvcc --use_fast_math --std=c++17 -o bmm_smem bmm_smem.cu
./bmm_smem 0