#!/bin/sh

clang-format -i *.cu *.h

# rm -rf bmm
# nvcc --use_fast_math --std=c++17 -o bmm bmm.cu
# ./bmm 3

rm -rf bmm_packedB
nvcc --use_fast_math --std=c++17 -o bmm_packedB bmm_packedB.cu
./bmm_packedB 3