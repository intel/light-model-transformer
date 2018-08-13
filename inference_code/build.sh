#!/bin/bash
rm -f test

export MKLDNN_ROOT=../third_party/mkl-dnn
export LD_LIBRARY_PATH=$MKLDNN_ROOT/build/src:$LD_LIBRARY_PATH
g++ -std=c++11 Main.cpp `pkg-config --cflags --libs opencv` -I$MKLDNN_ROOT/include -L$MKLDNN_ROOT/build/src -lmkldnn -ldl -liomp5 -O2 -o test
