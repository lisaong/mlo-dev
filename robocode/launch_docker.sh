#!/usr/bin/env bash
docker run -it \
  -m=6g \
  -p 8888:8888 \
  -v $PWD:/code \
  lisaong/robocode-dev:1.0 \
  bash
