#!/bin/sh

clang-format -i *.cu
nvcc -o bmm bmm.cu -run

# nsys profile --stats=true ./bmm