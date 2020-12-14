#!/usr/bin/env bash
docker run -it \
  -m=6g \
  -v $PWD:/code \
  lisaong/robocode-dev:1.0 \
  bash

