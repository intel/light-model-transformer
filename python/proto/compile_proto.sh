#!/bin/bash

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# MODEL_MODIFIER_SRC=/data/sources/tf_graph_rewrite

pushd $(dirname $0)

TENSORFLOW_PROTO=$1

protoc -I$TENSORFLOW_PROTO -I. --python_out=../ --mypy_out=../ model_modifier/*.proto

popd
