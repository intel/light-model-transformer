#!/usr/bin/env python

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_extractor import PatternExtractor

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
import tensorflow as tf

from google.protobuf.message import Error as ProtoError

import argparse

import os
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract a pattern from a tensorflow model graph.')
    parser.add_argument('path')
    parser.add_argument('-s', '--seed-nodes', nargs='+', default=[],
                        help='List of space-separated node names to use as seed nodes.')
    parser.add_argument('-b', '--barrier-nodes', nargs='+', default=[],
                        help='List of space-separated node names to use as barrier nodes.')
    parser.add_argument('-B', '--barrier-ops', nargs='+', default=[],
                        help='List of space-separated Op names to use as barriers. '
                        'All nodes with this op will become barrier nodes.')
    parser.add_argument(
        '-f', '--function', help='Name of the function inside the graph_def to use. If not provided, the root graph_def will be used.')
    parser.add_argument('-m', '--meta-graph', action='store', type=int, default=0,
                        help='Which meta graph of the saved model should be used. Only valid when using the saved model format.')
    parser.add_argument('-l', '--log', choices=['DEBUG', 'INFO', 'WARN',
                        'ERROR', 'FATAL'], default='WARN', help='Logger verbosity level.')

    required_named_arguments = parser.add_argument_group(
        'required named arguments')
    required_named_arguments.add_argument('-o', '--output', action='store', required=True,
                                          default=None, help='File to save the pattern to.')

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig()
    logging.root.setLevel(args.log)
    log = logging.getLogger(f'{__name__}.extract_pattern')

    try:
        if os.path.isdir(args.path):
            if tf.saved_model.contains_saved_model(args.path):

                log.info(
                    'Provided directory potentially contains a saved model. Attempting to load the saved model graph...')
                saved_model = SavedModel()
                saved_model_path = os.path.join(args.path, 'saved_model.pb')

                with open(saved_model_path, 'rb') as f:
                    saved_model.ParseFromString(f.read())

                try:
                    graph = saved_model.meta_graphs[args.meta_graph].graph_def
                except IndexError as e:
                    log.error(f'Error while picking the meta graph: {e}')
                    exit(1)

                log.info('Saved model graph loaded successfully.')

            else:
                log.error('Provided directory does not contain a saved model.')
                exit(1)

        elif os.path.isfile(args.path):
            graph = GraphDef()
            log.info('Attempting to load the frozen model graph...')
            with open(args.path, 'rb') as f:
                graph.ParseFromString(f.read())
            log.info('Frozen model graph loaded successfully.')
        else:
            log.error('Provided input path is not a file or directory.')
            exit(1)

    except IOError as e:
        log.error(f'Error while opening the file: {e}')
        exit(1)
    except ProtoError as e:
        log.error(f'Error while decoding the model: {e}')
        exit(1)

    try:
        extractor = PatternExtractor(graph)
        pattern = extractor.extract(seed_nodes=args.seed_nodes, barrier_nodes=args.barrier_nodes,
                                    barrier_ops=args.barrier_ops, function_name=args.function)
    except NotImplementedError as e:
        log.error(f'Pattern extraction failed with error: {e}')
        exit(1)

    if pattern is None:
        log.warn('Pattern extraction failed without error. Check input parameters.')
        exit(1)

    try:
        with open(args.output, 'wb') as f:
            f.write(pattern.SerializePartialToString())
    except IOError as e:
        log.error(f'Error while opening the output file: {e}')
        exit(1)


if __name__ == '__main__':
    main()
