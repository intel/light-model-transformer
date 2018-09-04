#!/bin/bash
rm -f test

export MKLDNN_ROOT=../third_party/mkl-dnn
export LD_LIBRARY_PATH=$MKLDNN_ROOT/build/src:$LD_LIBRARY_PATH
g++ -std=c++11 Main.cpp -I$MKLDNN_ROOT/include -L$MKLDNN_ROOT/build/src -lmkldnn -ldl -O2 -o test
