FROM lisaong/robocode-dev:2.0 AS builder

USER root
RUN sudo apt-get update \
    && apt-get install -y doxygen

WORKDIR /code
RUN git clone -b doc/programming_model https://intelligentdevices.visualstudio.com/ELL/_git/robocode robocode
RUN pip install -r /code/robocode/doc/DSL/docgen/requirements.txt
RUN mkdir /code/robocode/build

WORKDIR /code/robocode/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -GNinja & \
    cmake --build . --target DSL_Sphinx install --parallel 4

FROM nginx:latest
WORKDIR /usr/share/nginx/html
COPY --from=builder /code/robocode/build/install/share/html html
COPY --from=builder /code/robocode/build/install/share/sphinx sphinx

EXPOSE 80