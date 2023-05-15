#!/bin/sh

clang-format -i *.cu *.h
nvcc --use_fast_math --std=c++17 -o bmm bmm.cu
./bmm
# nsys profile --stats=true ./bmm