FROM lisaong/robocode-dev:2.0 AS builder

ARG REPO_PAT

USER root
RUN sudo apt-get update \
    && apt-get install -y doxygen

WORKDIR /code
RUN git clone -b doc/programming_model https://${REPO_PAT}@intelligentdevices.visualstudio.com/ELL/_git/robocode robocode
RUN pip install -r /code/robocode/doc/DSL/docgen/requirements.txt
RUN mkdir /code/robocode/build

WORKDIR /code/robocode/build
RUN conan remote add robocode http://robocode-packages.eastus.cloudapp.azure.com:8081/artifactory/api/conan/RoboCode-Packages
RUN conan user -p RoboCode2020 -r robocode admin
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja
RUN cmake --build . --target DSL_sphinx install --parallel 4

FROM nginx:latest
WORKDIR /usr/share/nginx/html
COPY --from=builder /code/robocode/build/install/share/html html
COPY --from=builder /code/robocode/build/install/share/sphinx sphinx

EXPOSE 80