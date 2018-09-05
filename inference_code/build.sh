#!/bin/bash
rm -f test

export MKLDNN_ROOT=../third_party/mkl-dnn
export LD_LIBRARY_PATH=$MKLDNN_ROOT/build/src:$LD_LIBRARY_PATH
g++ -std=c++11 -O2 -I$MKLDNN_ROOT/include -L$MKLDNN_ROOT/build/src -lmkldnn -ldl Main.cpp -o test

# Build the image inference example
#g++ -std=c++11 -O2 -I$MKLDNN_ROOT/include -L$MKLDNN_ROOT/build/src -lmkldnn -lopencv_core -lopencv_imgproc -lopencv_highgui ImageInference.cpp -o test2
