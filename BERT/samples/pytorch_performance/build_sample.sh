#!/bin/bash

cd $(dirname $0)/../..

docker build -t bert-op-pytorch-demo  -f samples/pytorch_performance/Dockerfile ${@:1} .
