#!/bin/bash

cd ../..

docker build -f samples/tensorflow_performance/Dockerfile $@ .
