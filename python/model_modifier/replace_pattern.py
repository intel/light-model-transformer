#!/usr/bin/env python

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_replacer import PatternReplacer

from model_modifier.recipe_pb2 import Recipe

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel

import tensorflow as tf

import argparse
import logging
import os


def main():
    logging.basicConfig()
    logging.root.setLevel(logging.INFO)

    log = logging.getLogger(f'{__name__}.replace_pattern')

    parser = argparse.ArgumentParser(
        description='Replace a node pattern in a model with an optimized equivalent.'
    )
    parser.add_argument('source_model', type=str, help='Path to the model.')
    parser.add_argument('-r', '--recipe', type=str,
                        help='Path to the Recipe proto.')
    parser.add_argument('-o', '--output', type=str,
                        help='Path of the output proto.')
    parser.add_argument('-m', '--meta-graph', action='store', type=int, default=0,
                        help='Which meta graph of the saved model should be used. Only valid when using the saved model format.')

    args = parser.parse_args()

    if os.path.isdir(args.source_model):
        if tf.saved_model.contains_saved_model(args.source_model):
            log.info('Provided directory potentially contains a saved model. Attempting to load the saved model graph...')
            model = SavedModel()
            saved_model_path = os.path.join(
                args.source_model, 'saved_model.pb')
            with open(saved_model_path, 'rb') as f:
                model.ParseFromString(f.read())
            graph = model.meta_graphs[args.meta_graph].graph_def
            log.info('Saved model graph loaded successfully.')
        else:
            raise ValueError(
                'Provided directory does not contain a saved model.')
    else:
        model = GraphDef()
        graph = model
        log.info('Attempting to load the frozen model graph...')
        with open(args.source_model, 'rb') as f:
            graph.ParseFromString(f.read())
        log.info('Frozen model graph loaded successfully.')

    recipe = Recipe()
    with open(args.recipe, 'rb') as f:
        recipe.ParseFromString(f.read())

    pattern_replacer = PatternReplacer(graph)

    if pattern_replacer.replace(recipe):

        with open(args.output, 'wb') as f:
            f.write(model.SerializeToString())



if __name__ == '__main__':
    main()
