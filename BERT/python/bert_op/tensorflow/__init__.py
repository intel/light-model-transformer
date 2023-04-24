# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import transformers

from .bert_op_tensorflow import TFBertEncoderOp

transformers.models.bert.modeling_tf_bert.TFBertEncoder = TFBertEncoderOp
