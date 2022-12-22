# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.11.0)
project(BertOpt VERSION 0.1.0)

set(BACKENDS "TF" "PT" CACHE STRING "Backends for which to build the BertOp, by default both 'TF' (tensorflow) and 'PT' (pytorch)")

add_subdirectory(src)
add_subdirectory(tests)

# [WIP] by kchutkie
# add_subdirectory(python)
