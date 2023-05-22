#!/bin/sh

docker run -it --cap-add=SYS_ADMIN --gpus all \
    -v /home/onglisa/accera.triton:/accera.triton \
    -v /home/onglisa/mlo-dev:/mlo-dev \
    -u $(id -u):$(id -g) \
    cuda_12.1.1
