FROM continuumio/miniconda3:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential ninja-build ccache cmake

RUN apt-get install -y \
    libunwind-dev libopenblas-dev

RUN  apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /code
RUN conan remote add robocode http://robocode-packages.eastus.cloudapp.azure.com:8081/artifactory/api/conan/RoboCode-Packages & \
  conan user -p RoboCode2020 -r robocode admin
