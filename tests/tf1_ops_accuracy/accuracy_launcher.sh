#!/bin/bash

base_dir=$1
echo "Accuracy launcher for TF" $TF_VERSION
echo "Base dir: " $base_dir

pushd $(dirname $0)

export PYTHONPATH=$PYTHONPATH:$base_dir/bert_google

path_to_model=$base_dir/bert_model
path_to_modified_model=$base_dir/modified_bert_model
data_dir=$base_dir/download_glue/glue_data/MRPC
vocab_path=$path_to_model/vocab.txt

$Python3_EXECUTABLE accuracy.py $path_to_model $data_dir $path_to_bertop --vocab_file=$vocab_path --out_file=$out_file
$Python3_EXECUTABLE accuracy.py $path_to_modified_model $data_dir $path_to_bertop --vocab_file=$vocab_path --out_file=$out_file

popd
