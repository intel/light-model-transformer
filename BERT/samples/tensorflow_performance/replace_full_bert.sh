#!/bin/bash

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set -e

pushd $(dirname $0)

# This will read the graph from $2/saved_model.pb and save an optimized copy
# to $2/modified_saved_model.pb
python -m model_modifier.replace_pattern $2 \
    -r recipe.pb \
    -o $2/modified_saved_model.pb \

popd
