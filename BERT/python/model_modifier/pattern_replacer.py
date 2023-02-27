# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_locator import PatternLocator

from model_modifier.recipe_pb2 import Recipe

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework.function_pb2 import FunctionDef

from tensorflow.python.framework import ops

from typing import Iterable, Dict, Union

import logging


class PatternReplacer:
    def __init__(self, graph_def: GraphDef):
        self.log = logging.getLogger(f'{__name__}.PatternReplacer')

        self.__graph_def = graph_def
        self._locator: PatternLocator = PatternLocator()
        self.__default_graph : ops.Graph = ops.get_default_graph()

    @property
    def graph_def(self):
        return self.__graph_def

    def replace(self, recipe: Recipe) -> bool:
        self._validate_recipe(recipe)

        # If the pattern was found in any of the node collections, the replacement was overall successful.
        # A boolean result for each node collection is stored here.
        success_list: list[bool] = []

        graphs: Iterable[Union[GraphDef, FunctionDef]]
        graphs = [self.__graph_def]
        graphs.extend(self.__graph_def.library.function)

        graph: Union[GraphDef, FunctionDef]
        for graph in graphs:
            if isinstance(graph, FunctionDef):
                self.__is_function = True
                name = graph.signature.name
                node_collection = graph.node_def
            else:
                self.__is_function = False
                name = 'graph_def'
                node_collection = graph.node

            recipe_copy = Recipe()
            recipe_copy.CopyFrom(recipe)
            success, node_mapping = self._locator.locate(recipe_copy.source_pattern, node_collection)

            if not success :
                self.log.debug(
                    f'Failed to locate the specified pattern in the target graph {name}.')
                continue

            self.__node_collection = node_collection
            pattern_nodes = node_mapping.values()

            self._attach_target_node_inputs(recipe_copy, node_mapping)
            self._attach_target_node_outputs(
                pattern_nodes, recipe_copy.target_node)
            self._delete_nodes_by_name(pattern_nodes)
            self._add_node_to_graph(recipe_copy.target_node)

            function_outputs: Iterable[str] = graph.ret.values() if self.__is_function else []
            self._delete_dangling_nodes(function_outputs)

            self.log.info(f'Pattern located and replaced in {name}.')

            success_list.append(success)

        return any(success_list)

    def _validate_recipe(self, recipe: Recipe) -> None:
        pass

    def _add_node_to_graph(self, node: NodeDef):
        if(node.name  in [node.name for node in self.__node_collection]):
            raise ValueError('node name already in node collection')        
        self.__node_collection.append(node)

    def _attach_target_node_inputs(self, recipe: Recipe, node_mapping: Dict[str, str]) -> None:
        all_pattern_nodes = [*recipe.source_pattern.seed_nodes,
                             *recipe.source_pattern.internal_nodes]

        # Find pattern nodes that contain the specified inputs
        for i in range(len(recipe.target_node.input)):
            target_node_input_name = self._strip_node_name(recipe.target_node.input[i])
            for pattern_node in all_pattern_nodes:
                try:
                    # Find out whether this pattern node has the input we are looking for.
                    input_idx = [self._strip_node_name(input) for input in pattern_node.input].index(target_node_input_name)

                    # Find the node in the graph that corresponds to this pattern node
                    # and fetch its input_idx'th input. This is the input we need for the target node.
                    node_with_required_input = self._get_node_by_name(
                        node_mapping[pattern_node.name])
                    final_target_input_name = node_with_required_input.input[input_idx]

                    recipe.target_node.input[i] = final_target_input_name

                    self.log.debug(
                        f'Mapped pattern input {target_node_input_name} to {final_target_input_name} in the target graph.')

                    break

                # This pattern node does not have the input we need. This is fine, we just try the next one.
                except ValueError:
                    pass

            # None of the pattern nodes have the input specified byt the target node NodeDef.
            # This generally means the NodeDef in the Recipe is incorrect.
            else:
                raise ValueError(
                    f'Pattern input {recipe.target_node.input[i]} could not be mapped to any tensor in the target graph.')

    def _get_node_by_name(self, node_name: str) -> NodeDef:
        stripped_node_name = self._strip_node_name(node_name)
        return next(node for node in self.__node_collection if node.name == stripped_node_name)

    def _strip_node_name(self, node_name: str) -> str:
        return node_name.lstrip('^').split(':', 1)[0]

    def _node_to_input_name(self, node: NodeDef) -> str:
        if not self.__is_function:
            return node.name
        else:
            # Nodes with list outputs are not currently supported.
            output_idx = 0

            op_def = self.__default_graph._get_op_def(node.op)

            # Ops with multiple output args are not currently supported.
            if(len(op_def.output_arg) != 1):
                raise ValueError('Ops with multiple output args are not currently supported')

            return f'{node.name}:{op_def.output_arg[0].name}:{output_idx}'

    def _delete_nodes_by_name(self, node_names: Iterable[str]) -> None:
        # Protobuf lists don't implement __setitem__ and don't have a setter,
        # so we need to do some old school gymnastics
        i = 0
        while i < len(self.__node_collection):
            if self.__node_collection[i].name in node_names:
                self.log.debug(
                    f'Removing node {self.__node_collection[i].name} from the graph.')
                del self.__node_collection[i]
            else:
                i += 1

    def _attach_target_node_outputs(self, remap_from: Iterable[str], remap_to: NodeDef) -> None:

        if self.__is_function:
            # This should be taken from the node's OpDef, but it requires a call
            # to tf.load_op_library() for the BERT op.
            output_arg = 'encoded'
            # This value might be added to the recipe, but it is only needed if the BERT op starts producing more than
            # one output tensor at some point.
            output_idx = 0

            # We must follow the 'node:out:idx' scheme for nodes inside functions
            # (see https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/function.proto#L57)
            remap_name = f'{remap_to.name}:{output_arg}:{output_idx}'
        else:
            remap_name = remap_to.name

        def _is_control_dependency(input: str) -> bool:
            return input.startswith('^')

        for node in self.__node_collection:
            for i in range(len(node.input)):
                input = node.input[i]
                if self._strip_node_name(input) in remap_from:

                    if _is_control_dependency(input):
                        raise NotImplementedError(f'Encountered control dependency {input}. '
                            'Control dependencies are not supported by the PatternReplacer.')

                    node.input[i] = remap_name
                    self.log.debug(
                        f'Modified node {node.name}. Input {node.input[i]} remapped to {remap_name}.')
                        
    # will not work correctly if there is a chain of dangling nodes
    def _delete_dangling_nodes(self, function_outputs: Iterable[str] = []) -> None:
        dangling_nodes: Iterable[NodeDef] = []
        for node in self.__node_collection:
            for other_node in self.__node_collection:
                if node.name in [self._strip_node_name(input) for input in other_node.input]:
                    break

                if node.name in [self._strip_node_name(output) for output in function_outputs]:
                    break
            else:
                self.log.info(f'Found dangling node {node}.')
                dangling_nodes.append(node)
        
        self._delete_nodes_by_name([node.name for node in dangling_nodes])
