#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set -e

base_dir=$1
echo "Accuracy launcher for TF" $TF_VERSION
echo "Base dir: " $base_dir

tmpdir=$(mktemp -d)
echo "Copying $base_dir into temporary directory $tmpdir"
cp -R $base_dir/* $tmpdir/


pushd $(dirname $0)

# BERT-base

path_to_model=$tmpdir/fine_tuned
path_to_modified_model=$tmpdir/modified_fine_tuned

printf "${CXX_COMPILER}\t${path_to_model##*/}\t${TF_VERSION}\t-\t-\t" >> $out_file
$Python3_EXECUTABLE accuracy.py $path_to_model $path_to_bertop --out-file=$out_file

$Python3_EXECUTABLE -m model_modifier.configure_bert_op \
    $QUANTIZATION \
    $BFLOAT16 \
    --no-calibrate \
    --quant-factors-path=$QUANT_FACTORS_DIR/quant_factors_uncased_L-12_H-768_A-12.txt \
    $path_to_modified_model

printf "${CXX_COMPILER}\t${path_to_modified_model##*/}\t${TF_VERSION}\t${QUANTIZATION}\t${BFLOAT16}\t" >> $out_file
$Python3_EXECUTABLE accuracy.py $path_to_modified_model $path_to_bertop --out-file=$out_file


# BERT-large

path_to_model=$tmpdir/bert-large-128-seq-len
path_to_modified_model=$tmpdir/modified-bert-large-128-seq-len

printf "${CXX_COMPILER}\t${path_to_model##*/}\t${TF_VERSION}\t-\t-\t" >> $out_file
$Python3_EXECUTABLE accuracy.py $path_to_model $path_to_bertop --out-file=$out_file

$Python3_EXECUTABLE -m model_modifier.configure_bert_op \
    $QUANTIZATION \
    $BFLOAT16 \
    --no-calibrate \
    --quant-factors-path=$QUANT_FACTORS_DIR/quant_factors_uncased_L-24_H-1024_A-16.txt \
    $path_to_modified_model

printf "${CXX_COMPILER}\t${path_to_modified_model##*/}\t${TF_VERSION}\t${QUANTIZATION}\t${BFLOAT16}\t" >> $out_file
$Python3_EXECUTABLE accuracy.py $path_to_modified_model $path_to_bertop --out-file=$out_file

popd
