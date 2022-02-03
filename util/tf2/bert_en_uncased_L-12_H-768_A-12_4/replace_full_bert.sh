#!/bin/bash

set -e

tmpdir=$(mktemp -d)

trap "rm -r $tmpdir" EXIT

echo "Temporary directory: $tmpdir"

pushd $(dirname $0)

pattern=$tmpdir/pattern.pb

python -m model_modifier.extract_pattern $1 -o $pattern \
-s \
    model/bert_encoder/transformer/layer_11/output_layer_norm/batchnorm/add_1 \
-b \
    model/bert_encoder/dropout/Identity \
    model/bert_encoder/self_attention_mask/mul \
-B \
    ReadVariableOp \
    Const \
-m 0 \
-f __inference__wrapped_model_7206

recipe=$tmpdir/recipe.pb

python -m model_modifier.make_recipe \
    $pattern \
    fused_bert_node_def.pbtxt \
    $recipe

python -m model_modifier.replace_pattern $2 \
    -r $recipe \
    -o $2/modified_saved_model.pb

popd
