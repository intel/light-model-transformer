#!/usr/bin/env python

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
import tensorflow as tf

import argparse
import os
from typing import List


def flatten_nodes(graphs: List[GraphDef]) -> List[NodeDef]:
    # Flatten all nodes into a single iterable.
    # This means all nodes in all graphs, and in all function defs of each graph.
    return [node for graph in graphs for node in graph.node] + \
        [node for graph in graphs for function in graph.library.function for node in function.node_def]


def filter_by_op(nodes: List[NodeDef], op: str) -> List[NodeDef]:
    return [node for node in nodes if node.op == op]


def configure_bert_op_nodes(graphs: List[GraphDef], args: argparse.Namespace) -> None:

    nodes = flatten_nodes(graphs)
    bert_nodes = filter_by_op(nodes, 'Bert')

    print(
        f'Following Bert nodes will be updated: {[node.name for node in bert_nodes]}')
    for node in bert_nodes:
        if args.quantization is not None:
            node.attr['QuantizableDataType'].type = tf.qint8.as_datatype_enum if args.quantization else tf.float32.as_datatype_enum
            # node.attr['UseQuantization'].type = tf.qint8.as_datatype_enum if args.quantization else tf.float.as_datatype_enum
        if args.bfloat16 is not None:
            node.attr['NonQuantizableDataType'].type = tf.bfloat16.as_datatype_enum if args.bfloat16 else tf.float32.as_datatype_enum
            # node.attr['UseBFloat16'].type = tf.bfloat16.as_datatype_enum if args.bfloat16 else tf.float.as_datatype_enum


def main():
    parser = argparse.ArgumentParser(
        description='Configure attributes of BertOp nodes in a model.')

    parser.add_argument(
        'model', help='Saved model folder or a frozen model .pb file')

    parser.add_argument('-q', '--quantization', dest='quantization', action='store_true', default=None,
                        help='Use int8 quantization.')
    parser.add_argument('-Q', '--no-quantization', dest='quantization', action='store_false',
                        help='Do not use int8 quantization.')

    parser.add_argument('-b', '--bfloat16', dest='bfloat16', action='store_true', default=None,
                        help='Use BFloat16 in supported operations.')
    parser.add_argument('-B', '--no-bfloat16', dest='bfloat16', action='store_false',
                        help='Do not BFloat16 in supported operations.')

    parser.add_argument('-o', '--output', default=None,
                        help='Location of the output .pb. If not provided, the model will be modified in-place.')

    args = parser.parse_args()

    if os.path.isdir(args.model):
        saved_model_path = os.path.join(args.model, 'saved_model.pb')
        if not args.output:
            args.output = saved_model_path

        if tf.saved_model.contains_saved_model(args.model):
            print('Provided directory potentially contains a saved model. Attempting to load the saved model graph...')
            saved_model = SavedModel()
            with open(saved_model_path, 'rb') as f:
                saved_model.ParseFromString(f.read())
            graphs = [
                meta_graph.graph_def for meta_graph in saved_model.meta_graphs]
            print('Saved model graph loaded successfully.')
        else:
            raise ValueError(
                'Provided directory does not contain a saved model.')
        output_proto = saved_model
    else:
        if not args.output:
            args.output = args.model

        graph = GraphDef()
        print('Attempting to load the frozen model graph...')
        with open(args.model, 'rb') as f:
            graph.ParseFromString(f.read())
        print('Frozen model graph loaded successfully.')
        graphs = [graph]
        output_proto = graph

    configure_bert_op_nodes(graphs, args)

    with open(args.output, 'wb') as f:
        f.write(output_proto.SerializeToString())


if __name__ == '__main__':
    main()
