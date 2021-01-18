#!/usr/bin/env bash
docker run -it \
  -m=6g \
  -v $PWD:/code \
  lisaong/robocode-rtdserver:1.0