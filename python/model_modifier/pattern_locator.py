# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from model_modifier.pattern_pb2 import Pattern

from tensorflow.core.framework.node_def_pb2 import NodeDef

from queue import LifoQueue as Stack

from typing import Optional, Sequence, Dict, List, Tuple, Iterable

from copy import copy

import logging


class PatternLocator:
    def __init__(self):
        self.log = logging.getLogger(f'{__name__}.PatternLocator')

    def locate(self, pattern: Pattern, nodes: Sequence[NodeDef]) -> Tuple[bool, Dict[str, str]]:
        self.__pattern = pattern
        self.__nodes = nodes

        success, node_mapping = self._locate_rec(0, {})
        if success:
            self.log.debug(f'Pattern match found. Node mapping:')
            for ref_node, mapped_node in node_mapping.items():
                self.log.debug(f'   {ref_node} -> {mapped_node}')
        else:
            self.log.debug(f'Pattern match not found.')

        return success, node_mapping

    def _locate_rec(self, seed_node_idx: int, node_mapping: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:

        # We are past the last seed node, report success (with an empty node mapping for completeness).
        if seed_node_idx == len(self.__pattern.seed_nodes):
            return True, {}

        reference_seed_node = self.__pattern.seed_nodes[seed_node_idx]
        potential_seed_nodes = self._get_nodes_with_op(reference_seed_node.op)

        self.log.debug(
            f'Looking for seed node {seed_node_idx + 1} of {len(self.__pattern.seed_nodes)}.')
        i = -1
        for node in potential_seed_nodes:
            i += 1
            self.log.debug(
                f'Trying node {i} of {len(potential_seed_nodes)} potential seed nodes.')
            self.log.debug(
                f'Trying to fit seed node {seed_node_idx} of the pattern to node {node.name}.')
            success, node_mapping_for_this_seed = self._pattern_fits_reference(
                node, reference_seed_node, node_mapping)
            if success:
                # Found a match for this seed node.
                merged_node_mapping = {**node_mapping,
                                       **node_mapping_for_this_seed}
                self.log.debug(
                    f'Seed node {seed_node_idx} of the pattern matches the graph at node {node.name}.')

                # Now try to find a match for the next seed node, with the fuller node mapping constraints.
                success, node_mapping_for_next_seed = self._locate_rec(
                    seed_node_idx + 1, merged_node_mapping)

                if success:
                    # Found a match for the next seed node.
                    current_complete_node_mapping = {
                        **merged_node_mapping, **node_mapping_for_next_seed}
                    return True, current_complete_node_mapping
            else:
                # This node does not fit a reference seed node, try the next candidate.
                self.log.debug(
                    f'Seed node {seed_node_idx} of the pattern does not match the graph at node {node.name}.')
                continue

        self.log.debug(
            f'No node of type {self.__pattern.seed_nodes[seed_node_idx].op} matches the pattern.')
        return False, {}

    def _pattern_fits_reference(self, node: NodeDef, ref_node: NodeDef, node_mapping: Dict[str, str]) -> Tuple[bool, Dict[str, str]]:
        open_nodes: Stack[Tuple[NodeDef, NodeDef]] = Stack()
        open_nodes.put((node, ref_node))

        FAIL: Tuple[bool, Dict[str, str]] = (False, {})

        # Node mapping with new nodes added as they are encountered in this pattern.
        # We don't want to extend the original node mapping until we are sure this pattern fits the reference pattern,
        # so we make a shallow copy, add to that copy instead and return the copy if the entire pattern is a match.
        extended_node_mapping = copy(node_mapping)

        while not open_nodes.empty():
            current_node, current_ref_node = open_nodes.get()
            self._throw_if_have_control_dependencies(
                [current_node, current_ref_node])

            self.log.debug(
                f'Analyzing node {current_node.name} against ref node {current_ref_node.name}')

            # Check if current_ref_node was previously mapped to a target graph node.
            try:
                expected_name = extended_node_mapping[current_ref_node.name]

                # If it was previously mapped to this node, we don't need to expand it (we must have verified this part
                # of the pattern before).
                if current_node.name == expected_name:
                    self.log.debug(
                        f'Skipping this part of the pattern, since it was previously analyzed.')
                    continue
                # But if it was mapped to a different node, this pattern does not match the reference (one ref_node
                # cannot be assigned to two different nodes in the target graph).
                else:
                    self.log.debug(
                        f'Ref node is already mapped to node {expected_name}.')
                    return FAIL

            except KeyError:
                # If the ref_node was node previously mapped, keep going.
                self.log.debug(f'Ref node not encountered previously.')

            if self._are_nodes_matching(current_node, current_ref_node, extended_node_mapping):
                self.log.debug(
                    f'Node matches ref. Adding a new entry to extended node mapping.')
                extended_node_mapping[current_ref_node.name] = current_node.name

                # current_node and current_ref_node are guaranteed to have the same number of inputs by call to
                # self._are_nodes_matching(...).
                for i in range(len(current_node.input)):
                    current_ref_node_input = self._get_node_by_name(
                        current_ref_node.input[i], self.__pattern.internal_nodes)
                    current_node_input = self._get_node_by_name(
                        current_node.input[i], self.__nodes)

                    if current_ref_node_input is not None and current_node_input is not None:
                        self.log.debug(
                            f'Adding node {current_node_input.name} and ref node {current_ref_node_input.name} to the open stack.')
                        open_nodes.put(
                            (current_node_input, current_ref_node_input))
                    elif current_ref_node_input is None:
                        self.log.debug(
                            f'Node input {current_node.input[i]} corresponds to pattern input {current_ref_node.input[i]}')
                        continue
                    else:
                        self.log.debug(
                            f'Input node {current_node.input[i]} is missing from the node collection, and it is required to fit the reference.')
                        return FAIL

            # If we detected a mismatch, then this part of the graph does not fit the reference pattern.
            else:
                self.log.debug(f'Node does not match the ref node.')
                return FAIL

        return True, extended_node_mapping

    def _throw_if_have_control_dependencies(self, nodes: Iterable[NodeDef]) -> None:
        def _throw_if_has_control_dependency(node: NodeDef) -> None:
            if any([input.startswith('^') for input in node.input]):
                raise NotImplementedError(f'Encountered control dependency {input}. '
                    'Control dependencies are not supported by the PatternLocator')

        for node in nodes:
            _throw_if_has_control_dependency(node)

    def _are_nodes_matching(self, node: NodeDef, ref_node: NodeDef, node_mapping: Dict[str, str]) -> bool:

        conditions: List[bool] = []

        conditions.append(node.op == ref_node.op)   # Must be same op
        # Must be same datatype
        conditions.append(node.attr.get('T') == ref_node.attr.get('T'))
        # Must be same number of inputs
        conditions.append(len(node.input) == len(ref_node.input))
        # Must be same output shape
        conditions.append(node.attr.get('_output_shapes')
                          == ref_node.attr.get('_output_shapes'))

        return all(conditions)

    def _get_node_by_name(self, node_name: str, nodes: Sequence[NodeDef]) -> Optional[NodeDef]:
        stripped_node_name = self._strip_node_name(node_name)
        return next((node for node in nodes if node.name == stripped_node_name), None)

    def _strip_node_name(self, node_name: str) -> str:
        '''Node inputs can be in the form "node_name:tensor_name:index", e.g. "add:z:0". By stripping everything after 
        the first colon, we can use functions like get_node_by_name on node names and on node inputs the same way.'''
        return node_name.split(':', 1)[0]

    def _is_internal_node(self, node: NodeDef, pattern: Pattern) -> bool:
        return node in pattern.internal_nodes

    def _get_nodes_with_op(self, op: str) -> Sequence[NodeDef]:
        return [node for node in self.__nodes if node.op == op]
