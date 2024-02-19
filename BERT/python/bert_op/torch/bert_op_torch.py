# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

import transformers
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from typing import Union, Tuple, Optional


class BertEncoderOp(transformers.models.bert.modeling_bert.BertEncoder):
    """
        This is a BertEncoder module which uses a monolithic C++ op to perform the computation.
    """

    class BertEncoderScriptModule(torch.nn.Module):
        def __init__(self, max_position_embeddings, hidden_size, intermediate_size, num_hidden_layers, num_att_heads,
                     use_quantization, use_bfloat16, quantization_factors, calibrate_quant_factors,
                     params):
            super().__init__()
            self._initialized = False

            self.max_position_embeddings = max_position_embeddings
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_att_heads = num_att_heads
            self.use_quantization = use_quantization
            self.use_bfloat16 = use_bfloat16
            # Convert from numpy array to list, to make it work with torchscript
            self.quantization_factors: list[float] = list(quantization_factors)
            self.calibrate_quant_factors = calibrate_quant_factors
            self.params = params

            self.bert_op = torch.classes.bert_op.BertOp()

        def forward(self, hidden_states, attention_mask):
            if not self._initialized:
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.numel() // batch_size // self.hidden_size
                self.bert_op.configure(seq_len,
                                       self.hidden_size,
                                       self.intermediate_size,
                                       batch_size,
                                       self.num_hidden_layers,
                                       self.num_att_heads,
                                       self.use_quantization, self.use_bfloat16, self.calibrate_quant_factors)
                self.bert_op.initialize(self.params, self.quantization_factors)
                self._initialized = True

            return self.bert_op.forward(hidden_states, attention_mask)
        
        def get_quantization_factors(self):
            return self.bert_op.get_quantization_factors()

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if not hasattr(self.config, "use_quantization"):
            self.config.use_quantization = False
        if not hasattr(self.config, "use_bfloat16"):
            self.config.use_bfloat16 = False
        if not hasattr(self.config, "quantization_factors"):
            # If not provided, a dummy value lets TorchScript infer the type of the list:
            self.config.quantization_factors = [1.]
        if not hasattr(self.config, "calibrate_quant_factors"):
            self.config.calibrate_quant_factors = False

        self.scriptable_bert_op = BertEncoderOp.BertEncoderScriptModule(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.num_hidden_layers,
            self.config.num_attention_heads,
            self.config.use_quantization, self.config.use_bfloat16, self.config.quantization_factors,
            self.config.calibrate_quant_factors,
            [*self.parameters()])

        # Quant factor calibration only works in eager mode, so we only use `torch.jit.script`
        # if calibration mode is not enabled. Otherwise, we cannot access the `get_quantization_factors` method.
        if not self.config.calibrate_quant_factors:
            self.scriptable_bert_op = torch.jit.script(self.scriptable_bert_op)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        if attention_mask is None:
            raise ValueError("attention_mask is required.")

        # Masking of attention heads is not supported, so the mask can be either None,
        # [None]*num_attention_heads or [1]*num_attention_heads.
        if head_mask is not None and any(v is not None and v != 1 for v in head_mask):
            raise ValueError("head_mask is not supported.")

        if encoder_hidden_states is not None:
            raise ValueError("encoder_hidden_states is not supported.")

        if encoder_attention_mask is not None:
            raise ValueError("encoder_attention_mask is not supported.")

        if past_key_values is not None:
            raise ValueError("past_key_values is not supported.")

        if use_cache:
            raise ValueError("use_cache option is not supported.")

        if output_attentions:
            raise ValueError("output_attentions option is not supported.")

        if output_hidden_states:
            raise ValueError("output_hidden_states option is not supported.")

        output: torch.FloatTensor = self.scriptable_bert_op(
            hidden_states, attention_mask)

        if not return_dict:
            return (output,)
        else:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=output
            )

    def get_quantization_factors(self):
        return self.scriptable_bert_op.get_quantization_factors()
