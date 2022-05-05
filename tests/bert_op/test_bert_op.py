import unittest

import tensorflow as tf
import numpy as np

import sys
import os

from types import ModuleType

class BertOpHelper(object):
    """
    Helper class to create a BertOp node and execute it.
    The class will create a correct configuration of attributes and inputs for the BertOp given minimal input.
    The configuration can then be used directly for positive test cases, or any element can be tampered with before
    execution for negative test cases.
    """

    def __init__(self, *, lib: ModuleType, batch: int = 1, max_token_size: int = 128, num_weights: int = 192,
                 hidden_size: int = 768, num_attention_heads: int = 12, intermediate_size: int = 3072,
                 quantizable_datatype: tf.DType = tf.float32, non_quantizable_datatype: tf.DType = tf.float32):

        if num_weights % 16 != 0:
            raise ValueError('num_weighs must be a multiple of 16.')

        self.lib = lib
        self.batch = batch
        self.max_token_size = max_token_size
        self.num_weights = num_weights
        self.layers = self.num_weights // 16
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size 
        self.quantizable_datatype = quantizable_datatype
        self.non_quantizable_datatype = non_quantizable_datatype

        self.input = np.zeros(shape=(batch, max_token_size, hidden_size), dtype=np.float32)
        self.mask = np.zeros(shape=(batch, max_token_size, max_token_size), dtype=np.float32)
        self.init_weights()

    def init_weights(self):
        query_weights = np.ones(shape=(self.hidden_size, self.hidden_size), dtype=np.float32)
        query_bias = np.ones(shape=(1, self.hidden_size), dtype=np.float32)

        key_weights = np.ones(shape=(self.hidden_size, self.hidden_size), dtype=np.float32)
        key_bias = np.ones(shape=(1, self.hidden_size), dtype=np.float32)

        value_weights = np.ones(shape=(self.hidden_size, self.hidden_size), dtype=np.float32)
        value_bias = np.ones(shape=(1, self.hidden_size), dtype=np.float32)

        attention_dense_weights = np.ones(shape=(self.hidden_size, self.hidden_size), dtype=np.float32)
        attention_dense_bias = np.ones(shape=(1, self.hidden_size), dtype=np.float32)

        norm1_gamma = np.ones(shape=(1, self.hidden_size), dtype=np.float32)
        norm1_beta = np.ones(shape=(1, self.hidden_size), dtype=np.float32)

        intermediate_weights = np.ones(shape=(self.hidden_size, self.intermediate_size), dtype=np.float32)
        intermediate_bias = np.ones(shape=(1, self.intermediate_size), dtype=np.float32)

        output_weights = np.ones(shape=(self.intermediate_size, self.hidden_size), dtype=np.float32)
        output_bias = np.ones(shape=(1, self.hidden_size), dtype=np.float32)

        norm2_gamma = np.ones(shape=(1, self.hidden_size), dtype=np.float32)
        norm2_beta = np.ones(shape=(1, self.hidden_size), dtype=np.float32)
        
        # Re-use the same weights for each layer.
        self.weights = self.layers * [query_weights, query_bias,
                               key_weights, key_bias,
                               value_weights, value_bias,
                               attention_dense_weights, attention_dense_bias,
                               norm1_gamma, norm1_beta,
                               intermediate_weights, intermediate_bias,
                               output_weights, output_bias,
                               norm2_gamma, norm2_beta]


    def call(self):
        return self.lib.Bert(
            embedded=self.input,
            input_mask=self.mask,
            weights=self.weights,
            QuantizableDataType=self.quantizable_datatype,
            NonQuantizableDataType=self.non_quantizable_datatype
        )


class BertOpTestCase(unittest.TestCase):
    def setUp(self):
        lib_path = os.environ.get("BERT_OP_PATH")
        if lib_path is None:
            raise RuntimeError("Missing environment variable BERT_OP_PATH")
        self.lib = tf.load_op_library(lib_path)

    def tearDown(self):
        # Workaround to force TF runtime to clear the kernel cache,
        # idea taken from https://github.com/tensorflow/tensorflow/issues/19671.
        # Otherwise we will get the "Batch size changed unexpectedly" error
        # from the BertOp in batched input tests.
        # Implementing dynamic batch size will make this workaround obsolete.
        tf.random.set_seed(1)

class TestBertOpDefault(BertOpTestCase):
    def test_default_valid_configuration(self):
        b = BertOpHelper(lib=self.lib) # A default valid FP32 configuration
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
        b = BertOpHelper(lib=self.lib, quantizable_datatype=tf.qint8, non_quantizable_datatype=tf.bfloat16)
        b.call()

    def test_wrong_quantization_dtype(self):
        b = BertOpHelper(lib=self.lib)
        b.quantizable_datatype = tf.int16 # Set to any incorrect dtype (i.e. not in [tf.float32, tf.qint8])
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

    def test_wrong_bf16_datatype(self):
        b = BertOpHelper(lib=self.lib)
        b.non_quantizable_datatype = tf.int16 # Set to any incorrect dtype (i.e. not in [tf.float32, tf.bfloat16])
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

class TestBertOpInputShapes(BertOpTestCase):

    def test_batch_input(self):
        b = BertOpHelper(lib=self.lib, batch=32)
        b.call()

    def test_embedded_wrong_number_of_dims(self):
        b = BertOpHelper(lib=self.lib)
        
        b.input = np.zeros(shape=(1, b.batch, b.max_token_size, b.hidden_size), dtype=np.float32)
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg="BertOp should fail for input tensors with more than 3 dimensions."):
            b.call()

        b.input = np.zeros(shape=(b.hidden_size), dtype=np.float32)
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg="BertOp should fail for input tensors with fewer than 2 dimensions."):
            b.call()

class TestBertOpMaskShapes(BertOpTestCase):
    
    def test_mask_wrong_number_of_dims(self):
        b = BertOpHelper(lib=self.lib)
        
        b.mask = np.zeros(shape=(1, b.batch, b.max_token_size, b.max_token_size), dtype=np.float32)
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg="BertOp should fail for mask tensors with more than 3 dimensions."):
            b.call()

        b.input = np.zeros(shape=(b.max_token_size, b.max_token_size), dtype=np.float32)
        with self.assertRaises(tf.errors.InvalidArgumentError,
                               msg="BertOp should fail for input tensors with fewer than 3 dimensions."):
            b.call()
