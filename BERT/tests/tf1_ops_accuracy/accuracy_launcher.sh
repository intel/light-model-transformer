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

path_to_model=$tmpdir/bert_model
path_to_modified_model=$tmpdir/modified_bert_model
data_dir=$tmpdir/download_glue/glue_data/MRPC
vocab_path=$path_to_model/vocab.txt

export PYTHONPATH=$PYTHONPATH:$base_dir/bert_google

pushd $(dirname $0)

printf "${CXX_COMPILER}\t${path_to_model##*/}\t${TF_VERSION}\t-\t-\t" >> $out_file
$Python3_EXECUTABLE accuracy.py --model_path=$path_to_model --data_dir=$data_dir --op_path=$path_to_bertop --vocab_file=$vocab_path --out_file=$out_file

$Python3_EXECUTABLE -m model_modifier.configure_bert_op $QUANTIZATION $BFLOAT16 $path_to_modified_model

printf "${CXX_COMPILER}\t${path_to_modified_model##*/}\t${TF_VERSION}\t${QUANTIZATION}\t${BFLOAT16}\t" >> $out_file
$Python3_EXECUTABLE accuracy.py --model_path=$path_to_modified_model --data_dir=$data_dir --op_path=$path_to_bertop --vocab_file=$vocab_path --out_file=$out_file

popd
