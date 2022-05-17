# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import unittest

import tensorflow as tf
import numpy as np

from enum import Enum
import sys
import os

from types import ModuleType
from typing import List

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

    def __init__(self, *, lib: ModuleType, batch: int = 1, max_token_size: int = 128, num_weights: int = 192,
                 hidden_size: int = 768, num_attention_heads: int = 12, intermediate_size: int = 3072,
                 quantizable_datatype: tf.DType = tf.float32, non_quantizable_datatype: tf.DType = tf.float32,
                 hidden_act: str = 'gelu_tanh', format: TensorFormat = TensorFormat.TF2, reuse_weights: bool = True):

        if num_weights % 16 != 0:
            raise ValueError('num_weighs must be a multiple of 16.')

        if hidden_size % num_attention_heads != 0:
            raise ValueError("HiddenSize must be a multiple of NumAttentionHeads")

        self.lib = lib
        self.batch = batch
        self.max_token_size = max_token_size
        self.num_weights = num_weights
        self.layers = self.num_weights // 16
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.quantizable_datatype = quantizable_datatype
        self.non_quantizable_datatype = non_quantizable_datatype
        self.hidden_act = hidden_act
        self.reuse_weights = reuse_weights
        self.tensor_format = format

        if self.tensor_format == TensorFormat.TF1:
            self.input = np.zeros(
                shape=(batch * max_token_size, hidden_size), dtype=np.float32)
            self.mask = np.zeros(shape=(batch, 1,
                                max_token_size), dtype=np.float32)
        elif self.tensor_format == TensorFormat.TF2:
            self.input = np.zeros(
                shape=(batch, max_token_size, hidden_size), dtype=np.float32)
            self.mask = np.zeros(shape=(batch, max_token_size,
                                max_token_size), dtype=np.float32)
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
            HiddenAct=self.hidden_act
        )


class BertOpTestCase(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, "lib"):
            lib_path = os.environ.get("BERT_OP_PATH")
            if lib_path is None:
                raise RuntimeError("Missing environment variable BERT_OP_PATH")
            self.lib = tf.load_op_library(lib_path)

    def tearDown(self):
        self.reset_tf_runtime()
    
    def reset_tf_runtime(self):
        # Workaround to force TF runtime to clear the kernel cache,
        # idea taken from https://github.com/tensorflow/tensorflow/issues/19671.
        # Otherwise we will get the "Batch size changed unexpectedly" error
        # from the BertOp in batched input tests.
        # Implementing dynamic batch size will make this workaround obsolete.
        tf.random.set_seed(1)


class TestBertOpDefault(BertOpTestCase):
    def test_non_batch_input(self):
        b = BertOpHelper(lib=self.lib)  # A default valid FP32 configuration
        b.call()

    def test_batch_input(self):
        b = BertOpHelper(lib=self.lib, batch=32)
        b.call()
    
    def test_non_batch_input_tf1(self):
        b = BertOpHelper(lib=self.lib, format=TensorFormat.TF1)
        b.call()
    
    def test_batch_input_tf1(self):
        b = BertOpHelper(lib=self.lib, batch=32, format=TensorFormat.TF1)
        b.call()


class TestBertOpAttributes(BertOpTestCase):
    def test_quantization(self):
        b = BertOpHelper(lib=self.lib, quantizable_datatype=tf.qint8)
        b.call()

    @unittest.skip("Can only be enabled on BF16-capable machines.")
    def test_bfloat16(self):
        b = BertOpHelper(lib=self.lib, non_quantizable_datatype=tf.bfloat16)
        b.call()

    @unittest.skip("Can only be enabled on BF16-capable machines")
    def test_quantization_bfloat16(self):
        b = BertOpHelper(lib=self.lib, quantizable_datatype=tf.qint8,
                         non_quantizable_datatype=tf.bfloat16)
        b.call()

    def test_wrong_quantization_dtype(self):
        b = BertOpHelper(lib=self.lib)
        # Set to any incorrect dtype (i.e. not in [tf.float32, tf.qint8])
        b.quantizable_datatype = tf.int16
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

    def test_wrong_bf16_datatype(self):
        b = BertOpHelper(lib=self.lib)
        # Set to any incorrect dtype (i.e. not in [tf.float32, tf.bfloat16])
        b.non_quantizable_datatype = tf.int16
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

    def test_wrong_hidden_act(self):
        b = BertOpHelper(lib = self.lib, hidden_act='invalid_value')
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

class TestBertOpEmbeddings(BertOpTestCase):
    def test_embedded_wrong_number_of_dims(self):
        b = BertOpHelper(lib=self.lib)

        b.input = np.zeros(
            shape=(1, b.batch, b.max_token_size, b.hidden_size), dtype=np.float32)
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg="BertOp should fail for input tensors with more than 3 dimensions."):
            b.call()

        b.input = np.zeros(shape=(b.hidden_size), dtype=np.float32)
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg="BertOp should fail for input tensors with fewer than 2 dimensions."):
            b.call()
    
    def test_embedded_invalid_shape(self):
        b = BertOpHelper(lib=self.lib)
        invalid_shape=(b.batch, b.max_token_size + 1, b.hidden_size)
        b.input = np.zeros(shape=invalid_shape, dtype=np.float32)
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

class TestBertOpMask(BertOpTestCase):
    def test_mask_wrong_number_of_dims(self):
        b = BertOpHelper(lib=self.lib)

        b.mask = np.zeros(shape=(1, b.batch, b.max_token_size,
                          b.max_token_size), dtype=np.float32)
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg="BertOp should fail for mask tensors with more than 3 dimensions."):
            b.call()

        b.mask = np.zeros(
            shape=(b.max_token_size, b.max_token_size), dtype=np.float32)
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg="BertOp should fail for mask tensors with fewer than 3 dimensions."):
            b.call()

    def test_mask_invalid_shape(self):
        b = BertOpHelper(lib=self.lib)
        invalid_shape=(b.batch, b.max_token_size + 1, b.max_token_size)
        b.mask = np.zeros(shape=invalid_shape, dtype=np.float32)
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)


class TestBertOpLayers(BertOpTestCase):
    def test_one_invalid_tensor_shape(self):
        self.invalid_shape = (1,)
        b = BertOpHelper(lib=self.lib, reuse_weights=False)
        for i in range(len(b.weights)):
            with self.subTest(i=i):
                # Backup a correct tensor and replace it with an invalid one
                self._tamper_test_case(b, i)
                with self.assertRaises(tf.errors.AbortedError,
                                       msg="BertOp should fail if any weight, bias etc. has incorrect dimensions."):
                    b.call()
                # Restore the correct tensor so that we only tamper with one tensor per subtest
                self._restore_test_case(b, i)
            self.reset_tf_runtime()

    def _tamper_test_case(self, b: BertOpHelper, i: int):
        self.backup = b.weights[i]
        b.weights[i] = np.ones(shape=self.invalid_shape, dtype=np.float32)

    def _restore_test_case(self, b: BertOpHelper, i: int):
        b.weights[i] = self.backup

    def test_num_weights_not_multiple_of_16(self):
        b = BertOpHelper(lib=self.lib)
        b.weights.append(b.weights[0])  # This would be a valid tensor for the next layer,
                                        # but we make the layer incomplete on purpose.
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg=f"BertOp should fail if NumWeights is not divisible by 16. "
                               f"NumLayers was {b.layers}, and NumWeights was {len(b.weights)}."):
            b.call()

    def test_number_of_layers_other_than_12(self):
        for num_layers in [1, 2, 4, 8, 16, 32]:
            with self.subTest(NumLayers=num_layers):
                b = BertOpHelper(lib = self.lib, num_weights=16*num_layers)
                self.assertRaises(tf.errors.InvalidArgumentError, b.call)
                self.reset_tf_runtime()
