# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_pb2 import Pattern

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef

from queue import LifoQueue as Stack

import sys
import logging

from typing import Sequence, List, Optional, Iterable, MutableSet


class PatternExtractor:
    UNSUPPORTED_OPS = ['PartitionedCall', 'StatefulPartitionedCall']

    def __init__(self, graph_def: GraphDef):
        self.graph_def = graph_def
        self.log = logging.getLogger(f'{__name__}.PatternExtractor')


    def extract(self, seed_nodes: Sequence[str], barrier_nodes: Sequence[str], barrier_ops: Sequence[str], function_name: Optional[str] = None) -> Optional[Pattern]:
        if function_name is not None:
            self.log.info(f'Extracting the pattern from function {function_name}.')
            return self._pattern_in_function(seed_nodes, barrier_nodes, barrier_ops, function_name)
        else:
            self.log.info(f'Extracting the pattern from the graph_def.')
            return self._pattern_in_node_collection(seed_nodes, barrier_nodes, barrier_ops, self.graph_def.node)

    def _pattern_in_function(self, seed_nodes: Sequence[str], barrier_nodes: Sequence[str], barrier_ops: Sequence[str], function_name: str) -> Optional[Pattern]:
        try:
            function_defs = self.graph_def.library.function
            node_collection = next(
                func.node_def for func in function_defs if func.signature.name == function_name)
            return self._pattern_in_node_collection(seed_nodes, barrier_nodes, barrier_ops, node_collection)
        except StopIteration:
            self.log.error(f'No function named {function_name}.')
            return None

    def _pattern_in_node_collection(self, seed_nodes: Sequence[str], barrier_nodes: Sequence[str], barrier_ops: Sequence[str], nodes: Iterable[NodeDef]) -> Optional[Pattern]:
        barrier_nodes = self._mark_specified_ops_as_barrier_nodes(
            barrier_nodes, barrier_ops, nodes)

        try:
            self._validate_pattern_constraints(seed_nodes, barrier_nodes, nodes)
        except ValueError as e:
            self.log.error(f'Pattern constraints validation failed: {e}')
            return None

        pattern = self._initialize_pattern(seed_nodes, nodes)

        open_nodes: Stack[NodeDef] = Stack()
        for node in pattern.seed_nodes:
            open_nodes.put(node)

        inputs_encountered: MutableSet[str] = set()

        while not open_nodes.empty():
            node = open_nodes.get()

            # We've reached a 'barrier' node, stop exploring this branch and record this node as an input.
            if node.name in barrier_nodes:
                self.log.debug(f'Encountered barrier node {node.name}. The node will be recorded as a pattern input.')
                inputs_encountered.add(node.name)
                continue

            # Nodes with no inputs (Placeholder for example) are not supported.
            if not node.input:
                self.log.error(f'Reached node {node.name} of type {node.op} with no inputs, and it is not a barrier node.')
                return None

            if node.op in PatternExtractor.UNSUPPORTED_OPS:
                self.log.error(f'Node {node.name} has unsupported op type {node.op}.')
                return None

            # Record this node as part of the pattern, unless it's already a seed node
            if node not in pattern.seed_nodes and node not in pattern.internal_nodes:
                self.log.debug(f'Recording pattern node {node.name}.')
                pattern.internal_nodes.append(node)

            for fanin_node in self._get_fanin_nodes(node, nodes):
                # If we have not not already expanded this node, add it to the open stack.
                if fanin_node not in pattern.internal_nodes and \
                   fanin_node not in pattern.seed_nodes:
                    self.log.debug(f'Adding node {fanin_node.name} to the processing stack.')
                    open_nodes.put(fanin_node)

        # Non-seed-node outputs can be allowed when generating the pattern.
        # For example, the BERT-base model on TFHub has outputs additional outputs
        # for each layer, but we don't need to specify each of them as a seed node
        # for the model modifier to work.
        # So, we have a check for this, but we don't need to be using it right now.
        # self._verify_all_outputs_are_seed_nodes(pattern, nodes)

        pattern.inputs.extend([n for n in inputs_encountered])
        pattern.inputs.sort()

        return pattern

    def _mark_specified_ops_as_barrier_nodes(self, barrier_nodes: Sequence[str], barrier_ops: Sequence[str], nodes: Iterable[NodeDef]) -> Sequence[str]:
        result: List[str] = list()
        for node in nodes:
            if node.op in barrier_ops or node.name in barrier_nodes:
                result.append(node.name)
        return result

    def _validate_pattern_constraints(self, seed_nodes: Sequence[str], barrier_nodes: Sequence[str], nodes: Iterable[NodeDef]) -> None:
        seed_set = set(seed_nodes)
        if len(seed_set) != len(seed_nodes):
            raise ValueError(f'Duplicate seed nodes were provided.')

        barrier_set = set(barrier_nodes)
        if len(barrier_set) != len(barrier_nodes):
            raise ValueError(f'Duplicate barrier nodes were provided.')

        common_set = seed_set & barrier_set
        if common_set:
            raise ValueError(
                f'Following nodes were provided as both seed and barrier nodes: {common_set}.'
            )

        node_names = [node.name for node in nodes]
        for node_name in seed_nodes:
            if node_name not in node_names:
                raise ValueError(
                    f'Seed Node {node_name} not found in node collection.')

        for node_name in barrier_nodes:
            if node_name not in node_names:
                raise ValueError(
                    f'Barrier Node {node_name} not found in node collection.')

    def _initialize_pattern(self, seed_nodes: Sequence[str], nodes: Iterable[NodeDef]) -> Pattern:
        pattern = Pattern()
        pattern.seed_nodes.extend(
            node for node in nodes if node.name in seed_nodes)
        return pattern

    def _get_fanin_nodes(self, node: NodeDef, nodes: Iterable[NodeDef]) -> Iterable[NodeDef]:
        input_node_names = [self._input_name_to_node_name(
            node_input) for node_input in node.input]

        fanin_nodes = [node for node in nodes if node.name in input_node_names]
        if (len(input_node_names) != len(fanin_nodes)):
            fanin_node_names = [node.name for node in fanin_nodes]
            difference = set.difference(set(input_node_names), set(fanin_node_names))
            raise ValueError(f'Could not find nodes {difference} listed as fanin of node {node.name}')

        return fanin_nodes

    def _input_name_to_node_name(self, input_name: str) -> str:
        '''
        Node inputs are named using the format: node_name:output_name:tensor_id,
        with the output_name being optional.

        Example: some_module/add:z:0.

        We extract the node name by taking the substring up to (not including) the first colon.
        '''

        def _is_control_dependency(input: str) -> bool:
            return input.startswith('^')

        if _is_control_dependency(input_name):
            raise NotImplementedError(f'Encountered control dependency {input_name}. '
                'Control dependencies are not supported by the pattern extractor.')

        return input_name.split(':', 1)[0]


    def _verify_all_outputs_are_seed_nodes(self, pattern: Pattern, nodes: Iterable[NodeDef]) -> None:
        internal_node_names = [node.name for node in pattern.internal_nodes]
        for node in nodes:
            if node not in pattern.seed_nodes and node not in pattern.internal_nodes:
                for node_input in node.input:
                    input_node_name = self._input_name_to_node_name(node_input)
                    if input_node_name in internal_node_names:
                        raise RuntimeError(
                            f'Node {node.name} takes input from node {input_node_name}, which is an internal node of the pattern.')
