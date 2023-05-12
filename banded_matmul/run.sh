#!/bin/sh

clang-format -i *.cu *.h
nvcc -o bmm bmm.cu -run

# nsys profile --stats=true ./bmm