#!/usr/bin/env bash
docker build --build-arg REPO_PAT=PATPAT -t='lisaong/robocode-rtdserver:1.0' .
docker tag lisaong/robocode-rtdserver:1.0 lisaong/robocode-rtdserver:latest