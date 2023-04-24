# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import transformers

if transformers.is_tf_available():
    from . import tensorflow

if transformers.is_torch_available():
    from . import torch
