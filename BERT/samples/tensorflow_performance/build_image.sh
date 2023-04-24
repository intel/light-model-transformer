#!/bin/bash

cd $(dirname $0)/../..

docker build -f samples/tensorflow_performance/Dockerfile $@ .
