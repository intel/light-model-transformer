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

pattern=$tmpdir/pattern.pbtxt

python -m model_modifier.extract_pattern $1 -o $pattern \
-s \
    bert/encoder/layer_23/output/layer_normalization_48/add \
-b \
    bert/encoder/Reshape_1 \
    bert/encoder/Reshape \
    bert/encoder/strided_slice \
    bert/encoder/strided_slice_2 \
-B \
    Identity \
    Const \
    ReadVariableOp \
-m 0

recipe=$tmpdir/recipe.pb

python -m model_modifier.make_recipe $pattern fused_bert_node_def.pbtxt $recipe

python -m model_modifier.replace_pattern $2 -r $recipe -o $3

popd
