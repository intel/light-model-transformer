# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(bert_layer)

if("TF" IN_LIST BACKENDS)
    add_subdirectory(tf_op)
endif()

if("PT" IN_LIST BACKENDS)
    add_subdirectory(pytorch_op)
endif()
