#!/bin/bash

base_dir=$1
echo "Accuracy launcher for TF" $TF_VERSION
echo "Base dir: " $base_dir

pushd $(dirname $0)

path_to_model=$base_dir/fine_tuned
path_to_modified_model=$base_dir/modified_fine_tuned

if [ -n "$2" ]; then    
    printf "${2}\t" >> $out_file
fi
$Python3_EXECUTABLE accuracy.py $path_to_model $path_to_bertop --out-file=$out_file

if [ -n "$2" ]; then    
    printf "${2}\t" >> $out_file
fi
$Python3_EXECUTABLE accuracy.py $path_to_modified_model $path_to_bertop --out-file=$out_file

popd
