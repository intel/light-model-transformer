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
    bert_encoder/transformer/layer_23/output_layer_norm/batchnorm/add_1 \
-b \
    bert_encoder/dropout/Identity \
    bert_encoder/self_attention_mask/mul \
-B \
    ReadVariableOp \
    Const \
-m 0 \
-f __inference_bert_layer_call_and_return_conditional_losses_22694

recipe=$tmpdir/recipe.pb

python -m model_modifier.make_recipe \
    $pattern \
    fused_bert_node_def.pbtxt \
    $recipe

python -m model_modifier.replace_pattern $2 \
    -r $recipe \
    -o $2/modified_saved_model.pb

popd
