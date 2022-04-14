#!/usr/bin/env python

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_replacer import PatternReplacer
from model_modifier.recipe_pb2 import Recipe

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
import tensorflow as tf

from google.protobuf.message import Error as ProtoError

from typing import Tuple

import argparse
import logging
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Replace a node pattern in a model with an optimized equivalent.'
    )
    parser.add_argument('source_model', type=str, help='Path to the model.')
    parser.add_argument('-m', '--meta-graph', action='store', type=int, default=0,
                        help='Which meta graph of the saved model should be used. Only valid when using the saved model format.')
    parser.add_argument('-l', '--log', choices=['DEBUG', 'INFO', 'WARN',
                        'ERROR', 'FATAL'], default='WARN', help='Logger verbosity level.')

    required_named_arguments = parser.add_argument_group(
        'required named arguments')
    required_named_arguments.add_argument('-r', '--recipe', type=str, required=True,
                                          help='Path to the Recipe proto.')
    required_named_arguments.add_argument('-o', '--output', type=str, required=True,
                                          help='Path of the output proto.')
    return parser.parse_args()

def prepare_saved_model(args) -> Tuple[SavedModel, GraphDef]:
    log = logging.getLogger(f'{__name__}.replace_pattern')
    model = SavedModel()
    saved_model_path = os.path.join(
        args.source_model, 'saved_model.pb')

    with open(saved_model_path, 'rb') as f:
        model.ParseFromString(f.read())

    try:
        graph = model.meta_graphs[args.meta_graph].graph_def
    except IndexError as e:
        log.error(f'Error while loading the graph: {e}')
        exit(1)

    # For saved models, the GraphDef is one of the fields.
    # We return both. The graph reference will be modified,
    # then the whole SavedModel will be written to a file.
    return model, graph

def prepare_frozen_model(args) -> Tuple[GraphDef, GraphDef]:
    graph = GraphDef()
    with open(args.source_model, 'rb') as f:
        graph.ParseFromString(f.read())

    # For frozen models, the GraphDef IS the entire model.
    # We return it twice - once as the whole "model",
    # and once as the "graph" to be modified. This way,
    # we don't need separate logic to modify the frozen
    # model and save it.
    return graph, graph


def main():
    args = parse_args()

    logging.basicConfig()
    logging.root.setLevel(args.log)
    log = logging.getLogger(f'{__name__}.replace_pattern')

    try:
        if os.path.isdir(args.source_model):
            if tf.saved_model.contains_saved_model(args.source_model):
                log.info(
                    'Provided directory potentially contains a saved model. Attempting to load the saved model graph...')

                model, graph = prepare_saved_model(args)

                log.info('Saved model graph loaded successfully.')

            else:
                log.error('Provided directory does not contain a saved model.')
                exit(1)

        elif os.path.isfile(args.source_model):
            log.info('Attempting to load the frozen model graph...')

            model, graph = prepare_frozen_model(args)

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

    recipe = Recipe()
    with open(args.recipe, 'rb') as f:
        recipe.ParseFromString(f.read())

    pattern_replacer = PatternReplacer(graph)

    try:
        if pattern_replacer.replace(recipe):
            with open(args.output, 'wb') as f:
                f.write(model.SerializeToString())
        else:
            log.error('Failed to find and replace the pattern in the model.')
            exit(1)

    except NotImplementedError as e:
        log.error(f'Pattern replacement failed with error: {e}')
        exit(1)
    except IOError as e:
        log.error(f'Error while opening the output file: {e}')
        exit(1)


if __name__ == '__main__':
    main()
