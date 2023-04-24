#!/bin/bash

cd $(dirname $0)/../..

docker build -f samples/tensorflow_no_model_modifier_performance/Dockerfile $@ .
