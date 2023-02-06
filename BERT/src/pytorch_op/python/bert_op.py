import torch

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from typing import Union, Tuple, Optional

from os import getenv

# Placeholder torch.nn.Module classes with no `forward` function are used here
# just to replicate the layout of the HuggingFace's BertEncoder parameters.
# This way we can load the weights from the original model without editing
# the state_dict in any way.
#
# There might be a cleaner way to do this, but this was the least invasive
# way I (krzychut) could find at the time.

class _BertSelfAttentionWeights(torch.nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type != "absolute":
            raise ValueError(f"Only absolute embeddings are supported.")
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(config.hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(config.hidden_size, self.all_head_size)


class _BertSelfOutputWeights(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

class _BertAttentionWeights(torch.nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = _BertSelfAttentionWeights(config, position_embedding_type=position_embedding_type)
        self.output = _BertSelfOutputWeights(config)

class _BertIntermediateWeights(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.hidden_act = config.hidden_act

class _BertOutputWeights(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

class _BertLayerWeights(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = _BertAttentionWeights(config)
        self.intermediate = _BertIntermediateWeights(config)
        self.output = _BertOutputWeights(config)

class BertEncoderOp(torch.nn.Module):
    """
        This is a BertEncoder module which uses a monolithic C++ op to perform the computation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = torch.nn.ModuleList([_BertLayerWeights(config) for _ in range(config.num_hidden_layers)])
        self.bert_op = torch.classes.bert_op.BertOp()
        self._weights_initialized = False

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

        if not self._weights_initialized:
            self.config.batch_size = hidden_states.shape[0]
            if not hasattr(self.config, "use_quantization"):
                self.config.use_quantization = False
            if not hasattr(self.config, "use_bfloat16"):
                self.config.use_bfloat16 = False
            if not hasattr(self.config, "quantization_factors"):
                self.config.quantization_factors = []
            self.bert_op.configure(self.config.max_position_embeddings,
                                   self.config.hidden_size,
                                   self.config.intermediate_size,
                                   self.config.batch_size, # batch size
                                   self.config.num_hidden_layers,
                                   self.config.use_quantization, self.config.use_bfloat16, False)
            self.bert_op.initialize([*self.parameters()], self.config.quantization_factors)
            self._weights_initialized = True

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

        output: torch.FloatTensor = self.bert_op.forward(hidden_states, attention_mask)
        
        if not return_dict:
            return (output,)
        else:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=output
            )

    def get_quantization_factors(self):
        return self.bert_op.get_quantization_factors()
# This will be moved to somewhere like the __init__.py of the package.
# The library path will obviously not be hard-coded.
import transformers
transformers.models.bert.modeling_bert.BertEncoder = BertEncoderOp

class EnvVariableNotSetError(Exception):
    pass

library_path = getenv('BERT_OP_LIB')
if library_path is None:
    raise EnvVariableNotSetError("Variable 'BERT_OP_LIB' must be set and point to the BertOp pytorch .so") 
torch.classes.load_library(library_path)
