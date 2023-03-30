# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import transformers
import torch
from bert_op.bert_op_torch import BertEncoderOp

transformers.models.bert.modeling_bert.BertEncoder = BertEncoderOp

from os import getenv

class EnvVariableNotSetError(Exception):
    pass

__library_path = getenv('BERT_OP_PT_LIB')
if __library_path is None:
    raise EnvVariableNotSetError(
        "Variable 'BERT_OP_PT_LIB' must be set and point to the BertOp pytorch .so")
torch.classes.load_library(__library_path)

del getenv
del transformers
del torch
