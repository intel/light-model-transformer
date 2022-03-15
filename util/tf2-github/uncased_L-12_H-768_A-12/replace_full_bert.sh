#!/bin/bash

set -e

tmpdir=$(mktemp -d)

trap "rm -r $tmpdir" EXIT

echo "Temporary directory: $tmpdir"

pushd $(dirname $0)

pattern=$tmpdir/pattern.pb

python -m model_modifier.extract_pattern $1 -o $pattern \
-s \
    bert/encoder/layer_11/output/layer_normalization_24/add \
-b \
    bert/encoder/strided_slice_2 \
    bert/encoder/strided_slice \
    bert/encoder/Reshape \
    bert/encoder/Reshape_1 \
-B \
    ReadVariableOp \
    Const \
    Identity

recipe=$tmpdir/recipe.pb

python -m model_modifier.make_recipe \
    $pattern \
    fused_bert_node_def.pbtxt \
    $recipe

python -m model_modifier.replace_pattern $2 \
    -r $recipe \
    -o modified_saved_model.pb

popd
