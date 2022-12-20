# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import unittest

import tensorflow as tf
import numpy as np

import os

from model_modifier.bert_op_helper import BertOpHelper, TensorFormat
from model_modifier import BERT_OP_ENV_VAR


class BertOpTestCase(unittest.TestCase):
    def setUp(self):
        if not hasattr(self, "lib"):
            lib_path = os.environ.get(BERT_OP_ENV_VAR)
            if lib_path is None:
                raise RuntimeError(f"Missing environment variable {BERT_OP_ENV_VAR}")
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

    def test_bert_large(self):
        b = BertOpHelper(lib=self.lib, max_token_size=512, num_weights=384, hidden_size=1024, num_attention_heads=16,
                         intermediate_size=4096)
        b.call()


class TestBertOpMaxTokenSize(BertOpTestCase):
    def test_max_token_size(self):
        for max_token_size in [64, 128, 256, 512]:
            with self.subTest(i=max_token_size):
                b = BertOpHelper(lib=self.lib, max_token_size=max_token_size)
                b.call()
                self.reset_tf_runtime()

    def test_invalid_max_token_size(self):
        b = BertOpHelper(lib=self.lib)
        b.input=np.zeros((b.batch, b.max_token_size - 1, b.hidden_size), dtype=np.float64)
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)


class TestBertOpQuantization(BertOpTestCase):
    def test_quantization(self):
        p = os.path.dirname(os.path.realpath(__file__))
        p = os.path.join(p, 'quant_factors_uncased_L-12_H-768_A-12.txt')
        b = BertOpHelper(lib=self.lib, quantizable_datatype=tf.qint8, quantization_factors_path=p)
        b.call()

    def test_wrong_quantization_dtype(self):
        b = BertOpHelper(lib=self.lib)
        # Set to any incorrect dtype (i.e. not in [tf.float32, tf.qint8])
        b.quantizable_datatype = tf.int16
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

    def test_quantization_factors_calibration(self):
        p = '/tmp/test_bert_op_TestBertOpQuantization_test_quantization_factors_calibration'
        # Cleanup previous executions, if any.
        try:
            os.remove(p)
        except FileNotFoundError: # FileNotFound is fine, any other exceptions should cause the test to abort
            pass

        b = BertOpHelper(lib=self.lib, calibrate_quant_factors=True, quantization_factors_path=p)
        b.call()
        with open(p, 'r') as f:
            text = f.read()
            text = text.strip()
            floats = [float(t) for t in text.split()]
            self.assertEqual(len(floats), b.layers*8)


    def test_quantization_invalid_quant_factors_file(self):
        p = os.path.dirname(os.path.realpath(__file__))
        p = os.path.join(p, 'quant_factors_uncased_L-12_H-768_A-12_INVALID.txt')
        b = BertOpHelper(lib=self.lib, quantizable_datatype=tf.qint8, quantization_factors_path=p)
        self.assertRaises(tf.errors.AbortedError, b.call)

    def test_quantization_quant_factors_file_not_exist(self):
        b = BertOpHelper(lib=self.lib, quantizable_datatype=tf.qint8, quantization_factors_path='file_does_not_exist')
        self.assertRaises(tf.errors.AbortedError, b.call)

    def test_quantization_with_calibration_mode_enabled(self):
        b = BertOpHelper(lib=self.lib, quantizable_datatype=tf.qint8, calibrate_quant_factors=True)
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)



class TestBertOpBFloat16(BertOpTestCase):
    @unittest.skip("Can only be enabled on BF16-capable machines.")
    def test_bfloat16(self):
        b = BertOpHelper(lib=self.lib, non_quantizable_datatype=tf.bfloat16)
        b.call()

    def test_wrong_bf16_datatype(self):
        b = BertOpHelper(lib=self.lib)
        # Set to any incorrect dtype (i.e. not in [tf.float32, tf.bfloat16])
        b.non_quantizable_datatype = tf.int16
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

    def test_bfloat16_with_calibration_mode_enabled(self):
        b = BertOpHelper(lib=self.lib, non_quantizable_datatype=tf.bfloat16, calibrate_quant_factors=True)
        self.assertRaises(tf.errors.InvalidArgumentError, b.call)

class TestBertOpQuantizationBFloat16(BertOpTestCase):
    @unittest.skip("Can only be enabled on BF16-capable machines")
    def test_quantization_bfloat16(self):
        b = BertOpHelper(lib=self.lib, quantizable_datatype=tf.qint8,
                         non_quantizable_datatype=tf.bfloat16)
        b.call()


class TestBertOpAttributes(BertOpTestCase):
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

    @unittest.skip("This test is no longer valid, "
                   "BertOp now supports models with different numbers of layers.")
    def test_number_of_layers_other_than_12(self):
        for num_layers in [1, 2, 4, 8, 16, 32]:
            with self.subTest(NumLayers=num_layers):
                b = BertOpHelper(lib = self.lib, num_weights=16*num_layers)
                self.assertRaises(tf.errors.InvalidArgumentError, b.call)
                self.reset_tf_runtime()
