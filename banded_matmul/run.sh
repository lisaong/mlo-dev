#!/bin/sh

nvcc -o bmm bmm.cu -run

# nsys profile --stats=true ./bmm