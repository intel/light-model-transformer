import transformers

import numpy as np
import tensorflow as tf

import os

from typing import Optional, Tuple, Union

class EnvVariableNotSetError(Exception):
    pass

class TFBertEncoderOp(transformers.models.bert.modeling_tf_bert.TFBertEncoder):
    def __init__(self, config: transformers.BertConfig, **kwargs):
        super().__init__(config, **kwargs)

        try:
            self.__bert_op_lib = os.environ['BERT_OP_LIB']
        except KeyError:
            raise EnvVariableNotSetError('BERT_OP_LIB variable is not set.')

        self.op = tf.load_op_library(self.__bert_op_lib).Bert
        self.__init_calls_done = False
        self.__num_init_calls = 0

        if not hasattr(self.config, 'use_quantization'):
            self.config.use_quantization = False
        if not hasattr(self.config, 'use_bfloat16'):
            self.config.use_bfloat16 = False
        if not hasattr(self.config, 'calibrate_quant_factors'):
            self.config.calibrate_quant_factors = False
        if not hasattr(self.config, 'quant_factors_path'):
            self.config.quant_factors_path = ''

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: Optional[tf.Tensor],
        encoder_attention_mask: Optional[tf.Tensor],
        past_key_values: Optional[Tuple[Tuple[tf.Tensor]]],
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[transformers.models.bert.modeling_tf_bert.TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:

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

        if past_key_values is not None and any([v is not None for v in past_key_values]):
            raise ValueError("past_key_values is not supported.")

        if use_cache:
            raise ValueError("use_cache option is not supported.")

        if output_attentions:
            raise ValueError("output_attentions option is not supported.")

        if output_hidden_states:
            raise ValueError("output_hidden_states option is not supported.")

        if training:
            raise ValueError("training option is not supported.")

        if not self.__init_calls_done:
            # During transformers.TFBertModel.from_pretrained, the model is called exactly twice with dummy inputs to
            # initialize the graph and make sure the restore ops are called. For these calls, we must fallback to
            # original behavior to let the weights initialize. On subsequent calls, we will feed these weights to the
            # BertOp.
            output = super().call(hidden_states,
                                  attention_mask,
                                  head_mask,
                                  encoder_hidden_states,
                                  encoder_attention_mask,
                                  past_key_values,
                                  use_cache,
                                  output_attentions,
                                  output_hidden_states,
                                  return_dict,
                                  training)

            self.__num_init_calls += 1
            if self.__num_init_calls == 2:
                self.__init_calls_done = True

        else:
            # We could add padding here instead of requiring padding from the tokenizer:

            # hidden_states = tf.pad(hidden_states,
            #                        [[0, 0],
            #                         [0, self.config.max_position_embeddings -
            #                             hidden_states.shape[1]],
            #                         [0, 0]])
            # attention_mask = tf.pad(attention_mask,
            #                         [[0, 0],
            #                          [0, 0],
            #                          [0, 0],
            #                          [0, self.config.max_position_embeddings - attention_mask.shape[3]]])
            batch_size = hidden_states.shape[0]
            self.config.seq_len = hidden_states.shape.num_elements() // batch_size // self.config.hidden_size

            hidden_states = self.op(
                embedded=hidden_states,
                input_mask=attention_mask,
                weights=self.weights,
                QuantizableDataType=tf.qint8 if self.config.use_quantization else tf.float32,
                NonQuantizableDataType=tf.bfloat16 if self.config.use_bfloat16 else tf.float32,
                HiddenSize=self.config.hidden_size,
                NumAttentionHeads=self.config.num_attention_heads,
                IntermediateSize=self.config.intermediate_size,
                MaxSequenceLength=self.config.seq_len,
                HiddenAct="gelu_tanh", # self.config.hidden_act is just "gelu"
                CalibrateQuantFactors=self.config.calibrate_quant_factors,
                QuantizationFactorsPath=self.config.quant_factors_path
            )

            if not return_dict:
                output = (hidden_states,)

            else:
                output = transformers.models.bert.modeling_tf_bert.TFBaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=hidden_states,
                )

        return output
