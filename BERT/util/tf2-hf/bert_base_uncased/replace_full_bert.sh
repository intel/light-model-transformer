#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set -e

tmpdir=$(mktemp -d)

trap "rm -r $tmpdir" EXIT

echo "Temporary directory: $tmpdir"

pushd $(dirname $0)

pattern=$tmpdir/pattern.pb

python -m model_modifier.extract_pattern $1 -o $pattern \
-s \
    bert/encoder/layer_._11/output/LayerNorm/batchnorm/add_1 \
-b \
    bert/embeddings/dropout/Identity \
    bert/Cast \
-B \
    ReadVariableOp \
    Const \
-m 0 \
-f __inference_call_7459

recipe=$tmpdir/recipe.pb

python -m model_modifier.make_recipe \
    $pattern \
    fused_bert_node_def.pbtxt \
    $recipe

python -m model_modifier.replace_pattern $2 \
    -r $recipe \
    -o $2/modified_saved_model.pb

popd
