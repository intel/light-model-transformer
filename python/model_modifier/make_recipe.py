#!/usr/bin/env python

# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_pb2 import Pattern
from model_modifier.recipe_pb2 import Recipe

from tensorflow.core.framework.node_def_pb2 import NodeDef

import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Utility script to create a Recipe protobuf given a Pattern and a NodeDef.'
    )

    parser.add_argument('-p', '--pattern', type=str,
                        help='Path to the pattern .pb file')
    parser.add_argument('-n', '--node-def', type=str,
                        help='Path to the NodeDef .pb file')
    parser.add_argument('-o', '--output', type=str,
                        help='Location of the Recipe .pb output file')

    args = parser.parse_args()

    try:
        with open(args.pattern, 'rb') as f:
            pattern = Pattern()
            pattern.ParseFromString(f.read())

        with open(args.node_def, 'rb') as f:
            node_def = NodeDef()
            node_def.ParseFromString(f.read())
    except Exception as e:
        print(f'Failed to load input data: {e}')
        exit(1)

    recipe = Recipe()
    recipe.source_pattern.CopyFrom(pattern)
    recipe.target_node.CopyFrom(node_def)

    try:
        with open(args.output, 'wb') as f:
            f.write(recipe.SerializeToString())
    except Exception as e:
        print(f'Failed to write the recipe file: {e}')
        exit(1)

if __name__ == '__main__':
    main()
