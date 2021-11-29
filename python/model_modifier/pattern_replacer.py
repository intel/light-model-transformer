# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_locator import PatternLocator

from model_modifier.recipe_pb2 import Recipe

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef

from typing import Iterable, Dict

import logging


class PatternReplacer:
    def __init__(self, graph_def: GraphDef):
        self.log = logging.getLogger(f'{__name__}.PatternReplacer')

        self.__graph_def = graph_def
        self._locator: PatternLocator = PatternLocator()

    @property
    def graph_def(self):
        return self.__graph_def

    def replace(self, recipe: Recipe) -> bool:
        self._validate_recipe(recipe)

        named_node_collections = {'graph_def': self.__graph_def.node}
        named_node_collections.update({str(func.signature.name): func.node_def for func in self.__graph_def.library.function})

        success_list: list[bool] = []

        for name, node_collection in named_node_collections.items():

            success, node_mapping = self._locator.locate(recipe.source_pattern, node_collection)

            if not success :
                self.log.info(
                    f'Failed to locate the specified pattern in the target graph in {name}.')
                continue

            self._add_node_to_graph(recipe.target_node)

            self._attach_target_node_inputs(recipe, node_mapping)

            pattern_nodes = node_mapping.values()

            self._attach_target_node_outputs(
                pattern_nodes, recipe.target_node.name)

            self._delete_nodes_by_name(pattern_nodes)

            self.log.info(f'Pattern located and replaced in {name}.')

            success_list.append(success)

        return any(success_list)

    def _validate_recipe(self, recipe: Recipe) -> None:
        pass

    def _add_node_to_graph(self, node: NodeDef):
        assert(node.name not in [node.name for node in self.__graph_def.node])
        self.__graph_def.node.append(node)

    def _attach_target_node_inputs(self, recipe: Recipe, node_mapping: Dict[str, str]) -> None:
        all_pattern_nodes = [*recipe.source_pattern.seed_nodes,
                             *recipe.source_pattern.internal_nodes]

        # Find pattern nodes that contain the specified inputs
        for i in range(len(recipe.target_node.input)):
            input_node = recipe.target_node.input[i]
            for node in all_pattern_nodes:
                try:
                    # Find out which subsequent input of the node this is (if any).
                    input_idx = list(node.input).index(input_node)

                    # Find the node in the graph that corresponds to this pattern node
                    # and get the same (input_idx'th) input.
                    target_node = self._get_node_by_name(
                        node_mapping[node.name])
                    target_input_name = target_node.input[input_idx]
                    recipe.target_node.input[i] = target_input_name

                    self.log.debug(
                        f'Mapped pattern input {input_node} to {target_input_name} in the target graph.')

                    break
                except ValueError:
                    pass
            else:
                raise ValueError(
                    f'Pattern input {recipe.target_node.input[i]} could not be mapped to any tensor in the target graph')

    def _get_node_by_name(self, name: str) -> NodeDef:
        return next(node for node in self.__graph_def.node if node.name == name)

    def _delete_nodes_by_name(self, node_names: Iterable[str]) -> None:
        # Protobuf lists don't implement __setitem__ and don't have a setter,
        # so we need to do some old school gymnastics
        i = 0
        while i < len(self.__graph_def.node):
            if self.__graph_def.node[i].name in node_names:
                self.log.debug(
                    f'Removing node {self.__graph_def.node[i].name} from the graph.')
                del self.__graph_def.node[i]
            else:
                i += 1

    def _attach_target_node_outputs(self, remap_from: Iterable[str], remap_to: str) -> None:
        for node in self.__graph_def.node:
            for i in range(len(node.input)):
                if node.input[i] in remap_from:
                    node.input[i] = remap_to
                    self.log.debug(
                        f'Modified node {node.name}. Input {node.input[i]} remapped to {remap_to}.')
