# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
import sys

def set_attr_shape(node, key, value):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(shape=tensor_shape.as_shape(value).as_proto()))
  except KeyError:
    pass

def set_attr_tensor(node, key, value, dtype, shape=None):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            value, dtype=dtype, shape=shape)))
  except KeyError:
    pass

def set_attr_int(node, key, value):
  try:
    node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))
  except KeyError:
    pass 

def create_node(op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
        new_node.input.extend([input_name])
    return new_node

def set_attr_dtype(node, key, value):
  try:
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(type=value.as_datatype_enum))
  except KeyError:
    pass

def create_constant_node(name, value, dtype, shape=None):
    node = create_node("Const", name, [])
    set_attr_dtype(node, "dtype", dtype)
    set_attr_tensor(node, "value", value, dtype, shape)
    return node

def create_placeholder(name):
    node = create_node('Placeholder', name, [])
    set_attr_dtype(node, "dtype", dtypes.int32)
    set_attr_shape(node, "shape", [None, 128])
    return node

################################################################################

def get_weight_names():
    weight_names = []

    for i in range(12): 
        name = 'bert/encoder/layer_' + str(i) + '/attention/self/query/kernel/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/attention/self/query/bias/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/attention/self/key/kernel/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/attention/self/key/bias/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/attention/self/value/kernel/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/attention/self/value/bias/read'
        weight_names.append(name)

        name = 'bert/encoder/layer_' + str(i) + '/attention/output/dense/kernel/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/attention/output/dense/bias/read'
        weight_names.append(name)

        name = 'bert/encoder/layer_' + str(i) + '/attention/output/LayerNorm/gamma/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/attention/output/LayerNorm/beta/read'
        weight_names.append(name)

        name = 'bert/encoder/layer_' + str(i) + '/intermediate/dense/kernel/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/intermediate/dense/bias/read'
        weight_names.append(name)

        name = 'bert/encoder/layer_' + str(i) + '/output/dense/kernel/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/output/dense/bias/read'
        weight_names.append(name)

        name = 'bert/encoder/layer_' + str(i) + '/output/LayerNorm/gamma/read'
        weight_names.append(name)
        name = 'bert/encoder/layer_' + str(i) + '/output/LayerNorm/beta/read'
        weight_names.append(name)

    return weight_names

def replace_sub_graph(graph_def, in_out_nodes, path_to_op):
    #tf.load_op_library('/data/sources/libraries.ai.bert/samples/bert2.so')
    tf.load_op_library(path_to_op)


    bert_input = 'bert/encoder/Reshape_1'
    bert_output = 'bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1'
    bert_mask = 'bert/encoder/Reshape'

    input_nodes = [bert_input, bert_mask]
    input_nodes.extend(get_weight_names())

    # Create the super Bert node
    bert_node = create_node('Bert', 'fused_bert', input_nodes)
    set_attr_dtype(bert_node, 'MaskT', tf.int64) # hard code, may need to change
    set_attr_int(bert_node, 'NumWeights', 16*12)
    graph_def.node.extend([bert_node])

    for node in graph_def.node:
        if not node.input:
            continue
        for i in range(len(node.input)):
            if str(node.input[i]) == bert_output:
                node.input[i] = bert_node.name
                print('**** Modified the input node of %s' % node.name)
                break

    print(in_out_nodes)
    graph_def = tf.compat.v1.graph_util.extract_sub_graph(graph_def, in_out_nodes)
    return graph_def