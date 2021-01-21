FROM lisaong/robocode-dev:2.0 AS builder

ARG REPO_PAT

USER root
RUN sudo apt-get update \
    && apt-get install -y clang-9

RUN sudo apt-get update \
    && apt-get install -y flex bison

WORKDIR /tmp
RUN git clone --branch Release_1_9_1 https://github.com/doxygen/doxygen.git
RUN mkdir doxygen/build

WORKDIR /tmp/doxygen/build
RUN cmake -G "Unix Makefiles" .. 
RUN make -j4
RUN sudo make install

WORKDIR /code
RUN git clone -b dev/onglisa/2967_prep_docs_framework https://${REPO_PAT}@intelligentdevices.visualstudio.com/ELL/_git/robocode robocode
RUN pip install -r /code/robocode/doc/DSL/requirements.txt
RUN mkdir /code/robocode/build

WORKDIR /code/robocode/build
RUN conan remote add robocode http://robocode-packages.eastus.cloudapp.azure.com:8081/artifactory/api/conan/RoboCode-Packages
RUN conan user -p RoboCode2020 -r robocode admin
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja -DROBOCODE_BUILD_DOCS=1
RUN cmake --build . --target dsl_docs install --parallel 4

FROM nginx:latest
WORKDIR /usr/share/nginx/html
COPY --from=builder /code/robocode/build/install/share/doc/ROBOCODE .

EXPOSE 80