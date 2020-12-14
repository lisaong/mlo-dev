FROM continuumio/miniconda3:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential ninja-build ccache

RUN apt-get install -y \
    libunwind-dev libopenblas-dev libncurses5

RUN  apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Compilers
ENV CC=gcc-8 \
    CXX=g++-8 \
    CMAKE_C_COMPILER=gcc-8 \
    CMAKE_CXX_COMPILER=g++-8

# Python libs
WORKDIR /tmp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# latest CMake
ADD https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1-Linux-x86_64.sh /tmp/install-cmake.sh
RUN mkdir /opt/cmake \
   & sh /tmp/install-cmake.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

# Pre-built LLVM
RUN conan remote add robocode http://robocode-packages.eastus.cloudapp.azure.com:8081/artifactory/api/conan/RoboCode-Packages & \
  conan user -p RoboCode2020 -r robocode admin

WORKDIR /code