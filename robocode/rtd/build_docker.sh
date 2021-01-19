#!/usr/bin/env bash
docker build --build-arg REPO_PAT=PATHERE -t=lisaong/robocode-rtdserver:1.2 -t=lisaong/robocode-rtdserver:latest --no-cache .