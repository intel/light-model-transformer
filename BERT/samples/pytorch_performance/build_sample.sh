#!/bin/bash

cd ../..

docker build -t bert-op-pytorch-demo  -f samples/pytorch_performance/Dockerfile ${@:1} .
