# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import glob
import sys
import os
import tensorflow as tf
import numpy as np
from utils import tf_utils, io_utils, saved_model_utils
from tensorflow.python.saved_model import tag_constants
import bert as bt

DT_FLOAT = 1
DT_DOUBLE = 2
DT_INT32 = 3
DT_UINT8 = 4
DT_INT16 = 5
DT_INT8 = 6
DT_STRING = 7
DT_COMPLEX64 = 8
DT_INT64 = 9
DT_BOOL = 10
DT_QINT8 = 11
DT_QUINT8 = 12
DT_QINT32 = 13
DT_BFLOAT16 = 14
DT_QINT16 = 15
DT_QUINT16 = 16
DT_UINT16 = 17
DT_COMPLEX128 = 18
DT_HALF = 19
DT_RESOURCE = 20
DT_VARIANT = 21
DT_UINT32 = 22
DT_UINT64 = 23


def get_shape_from_proto(shape_proto):
    return [dim.size for dim in shape_proto.dim]

def tf_dtype(t):
    if t == DT_FLOAT:
        return tf.float32
    elif t == DT_DOUBLE:
        return tf.float64
    elif t == DT_INT32:
        return tf.int32
    elif t == DT_UINT32:
        return tf.uint32
    elif t == DT_INT8:
        return tf.int8
    elif t == DT_UINT8:
        return tf.uint8
    elif t == DT_STRING:
        return tf.string

def get_dtype(t):
    if t == DT_FLOAT:
        return np.float32
    elif t == DT_INT32:
        return np.int32
    elif t == DT_UINT32:
        return np.uint32
    elif t == DT_INT8:
        return np.int8
    elif t == DT_UINT8:
        return np.uint8
    elif t == DT_STRING:
        return np.int32

def convert_savedmodel_to_savedmodel(model_dir, new_model_dir,path_to_op):
    # load saved_model which needs to be optimized
    sess = tf.compat.v1.Session()
    metagraph = tf.compat.v1.saved_model.loader.load(sess, [tag_constants.SERVING], model_dir)
    smi = saved_model_utils.SavedModelInfo(model_dir, [tag_constants.SERVING] )
    graph_def = smi._compatible_mgd.graph_def
    graph = tf.compat.v1.get_default_graph()

    input_dict = []
    output_dict = []
    output_list = []
    input_list = []
    sig_dict = {}
    
    
    for signature, ioconfig in metagraph.signature_def.items():
        sig_dict[signature] = {"in": {}, "out":{}}
        inputs_mapping = dict(ioconfig.inputs)
        outputs_mapping = dict(ioconfig.outputs)
        
        for k, v in inputs_mapping.items():
            input_dict.append(v.name)
            input_list.append(v.name.split(':')[-2])
            sig_dict[signature]["in"][k] = v.name
        
        for k, v in outputs_mapping.items():
            output_dict.append(v.name)
            output_list.append(v.name.split(':')[-2])
            sig_dict[signature]["out"][k] = v.name
        

    # relpace/optimize all subgraph
    constant_graph = graph_def

    in_out_nodes = output_list + input_list

    # graph, constant_graph = dl.replace_subgraph(constant_graph, in_out_nodes, model_dir)
    # graph, constant_graph = ec.replace_subgraph(graph, constant_graph,in_out_nodes)
    #graph, constant_graph = bc.replace_subgraph(graph, constant_graph, in_out_nodes)
    constant_graph = bt.replace_sub_graph(constant_graph, in_out_nodes,path_to_op)
    input_names = []
    input_types = []
    new_inputs = {}

    for node in constant_graph.node:
        if node.name in input_list:
            input_names.append(node.name)
            input_types.append(tf_dtype(node.attr['dtype'].type).as_datatype_enum)
            if isinstance(input_list, dict):
                new_inputs[input_list[node.name]] = node


    # save optimized model
    graph_def = constant_graph
    new_mgd = tf_utils.create_new_meta_graph_def(smi.origin_mgd, graph_def, new_inputs)
    io_utils.dump_new_saved_model(new_model_dir, model_dir, new_mgd, [tag_constants.SERVING])
    for filename in glob.iglob(model_dir + '/**/variables', recursive=True):
        os.system("cp -rf {} {}/".format(filename, new_model_dir))
        break


if __name__ == '__main__':
    debug_print_limit = 1
    debug_print = 0

    model_dir = sys.argv[1]
    path_to_op = sys.argv[2]
    new_model_dir = sys.argv[3]
    
    convert_savedmodel_to_savedmodel(model_dir, new_model_dir,path_to_op)
