#!/bin/sh

clang-format -i *.cu *.h

rm -rf bmm
nvcc --use_fast_math --std=c++17 -o bmm bmm.cu
