import tensorflow as tf
import numpy as np

from enum import Enum

from types import ModuleType
from typing import List

from model_modifier import TENSORS_PER_LAYER

class TensorFormat(Enum):
    TF1 = 1
    TF2 = 2


class BertOpHelper(object):
    """
    Helper class to create a BertOp node and execute it.
    The class will create a correct configuration of attributes and inputs for the BertOp given minimal input.
    The configuration can then be used directly for positive test cases, or any element can be tampered with before
    execution for negative test cases.
    """

    def __init__(self, *, lib: ModuleType, batch: int = 1, mask_type: tf.DType = tf.float32, max_token_size: int = 128,
                 num_weights: int = 12 * TENSORS_PER_LAYER, hidden_size: int = 768, num_attention_heads: int = 12, intermediate_size: int = 3072,
                 quantizable_datatype: tf.DType = tf.float32, non_quantizable_datatype: tf.DType = tf.float32,
                 hidden_act: str = 'gelu_tanh', calibrate_quant_factors: bool = False, quantization_factors_path: str = '',
                 format: TensorFormat = TensorFormat.TF2, reuse_weights: bool = True):

        if num_weights % TENSORS_PER_LAYER != 0:
            raise ValueError(f'num_weighs must be a multiple of {TENSORS_PER_LAYER}.')

        if hidden_size % num_attention_heads != 0:
            raise ValueError("HiddenSize must be a multiple of NumAttentionHeads")

        self.lib = lib
        self.batch = batch
        self.max_token_size = max_token_size
        self.num_weights = num_weights
        self.layers = self.num_weights // TENSORS_PER_LAYER
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.quantizable_datatype = quantizable_datatype
        self.non_quantizable_datatype = non_quantizable_datatype
        self.hidden_act = hidden_act
        self.calibrate_quant_factors = calibrate_quant_factors
        self.quantization_factors_path = quantization_factors_path
        self.reuse_weights = reuse_weights
        self.tensor_format = format

        mask_type = type(mask_type.as_numpy_dtype())
        if self.tensor_format == TensorFormat.TF1:
            self.input = np.zeros(
                shape=(batch * max_token_size, hidden_size), dtype=np.float32)
            self.mask = np.zeros(shape=(batch, 1,
                                max_token_size), dtype=mask_type)
        elif self.tensor_format == TensorFormat.TF2:
            self.input = np.zeros(
                shape=(batch, max_token_size, hidden_size), dtype=np.float32)
            self.mask = np.zeros(shape=(batch, max_token_size,
                                max_token_size), dtype=mask_type)
        else:
            raise ValueError("Invalid TensorFormat")

        self.init_weights()

    def init_weights(self):
        if self.tensor_format == TensorFormat.TF1:
            qkv_w = (self.hidden_size, self.hidden_size)
            qkv_b = (self.hidden_size,)

            att_w = (self.hidden_size, self.hidden_size)
            att_b = (self.hidden_size,)
        elif self.tensor_format == TensorFormat.TF2:
            qkv_w = (self.hidden_size, self.num_attention_heads, self.head_size)
            qkv_b = (self.num_attention_heads, self.head_size)

            att_w = (self.num_attention_heads, self.head_size, self.hidden_size)
            att_b = (self.hidden_size,)
        else:
            raise ValueError("Invalid TensorFormat.")

        gamma = (self.hidden_size,)
        beta = (self.hidden_size,)
        int_w = (self.hidden_size, self.intermediate_size)
        int_b = (self.intermediate_size,)
        out_w = (self.intermediate_size, self.hidden_size)
        out_b = (self.hidden_size,)

        query_weights = np.ones(
            shape=qkv_w, dtype=np.float32)
        query_bias = np.ones(shape=qkv_b, dtype=np.float32)

        key_weights = np.ones(
            shape=qkv_w, dtype=np.float32)
        key_bias = np.ones(shape=qkv_b, dtype=np.float32)

        value_weights = np.ones(
            shape=qkv_w, dtype=np.float32)
        value_bias = np.ones(shape=qkv_b, dtype=np.float32)

        attention_dense_weights = np.ones(
            shape=att_w, dtype=np.float32)
        attention_dense_bias = np.ones(
            shape=att_b, dtype=np.float32)

        norm1_gamma = np.ones(shape=gamma, dtype=np.float32)
        norm1_beta = np.ones(shape=beta, dtype=np.float32)

        intermediate_weights = np.ones(
            shape=int_w, dtype=np.float32)
        intermediate_bias = np.ones(
            shape=int_b, dtype=np.float32)

        output_weights = np.ones(
            shape=out_w, dtype=np.float32)
        output_bias = np.ones(shape=out_b, dtype=np.float32)

        norm2_gamma = np.ones(shape=gamma, dtype=np.float32)
        norm2_beta = np.ones(shape=beta, dtype=np.float32)

        if self.reuse_weights:
            self.weights = self.layers * [query_weights, query_bias,
                                          key_weights, key_bias,
                                          value_weights, value_bias,
                                          attention_dense_weights, attention_dense_bias,
                                          norm1_gamma, norm1_beta,
                                          intermediate_weights, intermediate_bias,
                                          output_weights, output_bias,
                                          norm2_gamma, norm2_beta]
        else:
            self.weights: List[np.ndarray] = []
            for _ in range(self.layers):
                self.weights.extend([query_weights.copy(), query_bias.copy(),
                                     key_weights.copy(), key_bias.copy(),
                                     value_weights.copy(), value_bias.copy(),
                                     attention_dense_weights.copy(), attention_dense_bias.copy(),
                                     norm1_gamma.copy(), norm1_beta.copy(),
                                     intermediate_weights.copy(), intermediate_bias.copy(),
                                     output_weights.copy(), output_bias.copy(),
                                     norm2_gamma.copy(), norm2_beta.copy()])

    def call(self):
        return self.lib.Bert(
            embedded=self.input,
            input_mask=self.mask,
            weights=self.weights,
            QuantizableDataType=self.quantizable_datatype,
            NonQuantizableDataType=self.non_quantizable_datatype,
            HiddenSize=self.hidden_size,
            NumAttentionHeads=self.num_attention_heads,
            IntermediateSize=self.intermediate_size,
            MaxSequenceLength=self.max_token_size,
            HiddenAct=self.hidden_act,
            CalibrateQuantFactors=self.calibrate_quant_factors,
            QuantizationFactorsPath=self.quantization_factors_path
        )
