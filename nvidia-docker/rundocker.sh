#!/bin/sh

docker run -it --cap-add=SYS_ADMIN --gpus all \
    -v /home/onglisa/accera.triton:/accera.triton \
    -u $(id -u):$(id -g) \
    cuda_12.1.1
