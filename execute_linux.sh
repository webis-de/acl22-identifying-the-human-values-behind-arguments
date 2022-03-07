#!/bin/bash

docker build -f Dockerfile -t acl22_values:no_cuda .

sudo docker run --rm -it --init --volume="$PWD:/app" acl22_values:no_cuda python main.py