#!/usr/bin/env python

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_extractor import PatternExtractor

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
import tensorflow as tf

import argparse

import os


def main():
    parser = argparse.ArgumentParser(
        description='Extract a pattern from a tensorflow model graph.')
    parser.add_argument('path')
    parser.add_argument('-s', '--seed-nodes', nargs='+', default=[],
                        help='List of space-separated node names to use as seed nodes.')
    parser.add_argument('-b', '--barrier-nodes', nargs='+', default=[],
                        help='List of space-separated node names to use as barrier nodes.')
    parser.add_argument('-B', '--barrier-ops', nargs='+', default=[],
                        help='List of space-separated Op names to use as barriers.')
    parser.add_argument(
        '-f', '--function', help='Name of the function inside the graph_def to use. If not provided, the root graph_def will be used.')
    parser.add_argument('-o', '--output', action='store',
                        default=None, help='File to save the pattern to.')
    parser.add_argument('-m', '--meta-graph', action='store', type=int, default=0,
                        help='Which meta graph of the saved model should be used. Only valid when using the saved model format.')
    args = parser.parse_args()

    if os.path.isdir(args.path):
        if tf.saved_model.contains_saved_model(args.path):
            print('Provided directory potentially contains a saved model. Attempting to load the saved model graph...')
            saved_model = SavedModel()
            saved_model_path = os.path.join(args.path, 'saved_model.pb')
            with open(saved_model_path, 'rb') as f:
                saved_model.ParseFromString(f.read())
            graph = saved_model.meta_graphs[args.meta_graph].graph_def
            print('Saved model graph loaded successfully.')
        else:
            raise ValueError(
                'Provided directory does not contain a saved model.')
    else:
        graph = GraphDef()
        print('Attempting to load the frozen model graph...')
        with open(args.path, 'rb') as f:
            graph.ParseFromString(f.read())
        print('Frozen model graph loaded successfully.')

    extractor = PatternExtractor(graph)
    pattern = extractor.extract(seed_nodes=args.seed_nodes, barrier_nodes=args.barrier_nodes,
                                barrier_ops=args.barrier_ops, function_name=args.function)
    if pattern is None:
        raise RuntimeError("Subgraph not found in graph")

    all_pattern_nodes = [node for node in pattern.seed_nodes] + \
        [node for node in pattern.internal_nodes]

    all_pattern_node_names = [node.name for node in all_pattern_nodes]
    all_pattern_nodes_set = set(all_pattern_node_names)
    if len(all_pattern_node_names) != len(all_pattern_nodes_set):
        print('Duplicate nodes found in pattern')
    else:
        print('No duplicates in pattern.')

    if args.output is None:
        print(pattern)
    else:
        with open(args.output, 'wb') as f:
            f.write(pattern.SerializePartialToString())


if __name__ == '__main__':
    main()
