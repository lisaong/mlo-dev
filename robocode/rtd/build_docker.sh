#!/usr/bin/env bash
docker build --build-arg REPO_PAT=PATHERE -t=lisaong/robocode-rtdserver:1.0 -t=lisaong/robocode-rtdserver:latest --no-cache .