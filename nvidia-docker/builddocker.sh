docker build . --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t cuda_12.1.1
