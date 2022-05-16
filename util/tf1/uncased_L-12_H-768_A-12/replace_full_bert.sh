#!/bin/bash

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set -e

tmpdir=$(mktemp -d)

echo "tmp dir: $tmpdir"

trap "rm -r $tmpdir" EXIT

echo "Temporary directory: $tmpdir"

pushd $(dirname $0)

pattern=$tmpdir/pattern.pb

python -m model_modifier.extract_pattern $1 -o $pattern \
-s \
    bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1 \
-b \
    bert/encoder/Reshape_1 \
    bert/encoder/Reshape \
    bert/encoder/strided_slice \
    bert/encoder/strided_slice_2 \
-B \
    Identity \
    Const \
-m 0

recipe=$tmpdir/recipe.pb

python -m model_modifier.make_recipe -p $pattern -n fused_bert_node_def.pb -o $recipe

python -m model_modifier.replace_pattern $2 -r $recipe -o $2/modified_saved_model.pb

popd
