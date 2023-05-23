#!/bin/sh

docker run -it --cap-add=SYS_ADMIN --gpus all \
    -v $HOME/accera.triton:/accera.triton \
    -v $HOME/mlo-dev:/mlo-dev \
    -u $(id -u):$(id -g) \
    cuda_12.1.1
