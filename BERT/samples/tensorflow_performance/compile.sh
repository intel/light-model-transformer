#!/bin/bash

set -e

. /opt/intel/oneapi/setvars.sh --force intel64

# Set up a build dir and compile
cd ../..
mkdir -p build && cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=icx \
    -DCMAKE_CXX_COMPILER=icpx \
    -DBACKENDS=TF
cmake --build . -j

# Compile the protobufs used by the model_modifier package
TENSORFLOW_PROTO=$(find /usr/local/lib/ -type d -wholename */dist-packages/tensorflow/include)
cd ../python/proto
protoc -I$TENSORFLOW_PROTO -I. --python_out=../ model_modifier/*.proto
