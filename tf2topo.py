#===============================================================================
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

import sys
import argparse
if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle
import os
from collections import OrderedDict, defaultdict
import tensorflow as tf
import numpy as np
from src.pb_nodes import get_consts
from src.utils import *
import re
import copy

logger = get_logger()

# Exception classes
class MergeException(Exception):
    pass

class MulCannotMergeException(MergeException):
    pass

class AddCannotMergeException(MergeException):
    pass

class BatchNormCannotMergeException(MergeException):
    pass

class ConvertException(Exception):
    pass

class FindNonconstException(ConvertException):
    pass

# Node wrapper
class Node(object):
    # node is a NodeDef object, refer to https://www.tensorflow.org/extend/tool_developers/
    def __init__(self, node):
        self._node = node
        self.name = node.name
        self.op = node.op
        self.input = node.input
        self.is_const = None
        self.layer_name = None
        self.attr = None

        # True means that the node is already merged to other nodes
        self.merged = False

        # Some nodes may be useless (like the input of 'Merge')
        self.is_useless = None

        # Input and output nodes (each element is in Node type) 
        self.input_nodes = []
        self.output_nodes = []

        # Output shape
        self.output_shape = None

        # Node calculation sequence, while following the seq can reach this node
        # Each element is a Node
        self.calc_seq = []

    def __repr__(self):
        return "<Name: %s  OP: %s>" % (self.name, self.op)


class ConvNode(Node):
    def __init__(self, node):
        super(ConvNode, self).__init__(node)
        self.data_format = "NHWC"
        self.have_bias = False
        self.conv_weights = None
        self.bias_weights = None
        self.pad = [-1, -1, -1, -1]

    def mul(self, multiplier):
        if len(multiplier.shape) != 1:
            logger.warning("Cannot merge mul node to {}, multiplier.shape={}".format(self.name, multiplier.shape))
            return False
        self.conv_weights = do_multiply(self, self.conv_weights, multiplier)
        return True

    def add_bias(self, bias):
        # TODO: check the bias
        if self.have_bias:
            for i in range(0, bias.shape[0]):
                self.bias_weights[i] += bias[i]
        else:
            self.bias_weights = bias
            self.have_bias = True

class DeConvNode(Node):
    def __init__(self, node):
        super(DeConvNode, self).__init__(node)
        self.data_format = "NHWC"
        self.have_bias = False
        self.conv_weights = None
        self.bias_weights = None
        self.pad = [-1, -1, -1, -1]
        self.output_shape = []

    def mul(self, multiplier):
        if len(multiplier.shape) != 1:
            logger.warning("Cannot merge mul node to {}, multiplier.shape={}".format(self.name, multiplier.shape))
            return False
        self.conv_weights = do_multiply(self, self.conv_weights, multiplier)
        return True

    def add_bias(self, bias):
        # TODO: check the bias
        if self.have_bias:
            for i in range(0, bias.shape[0]):
                self.bias_weights[i] += bias[i]
        else:
            self.bias_weights = bias
            self.have_bias = True

class FcNode(Node):
    def __init__(self, node):
        super(FcNode, self).__init__(node)
        self.have_bias = False
        self.fc_weights = None
        self.bias_weights = None

    def mul(self, multiplier):
        if len(multiplier.shape) != 1:
            logger.warning("Cannot merge mul node to {}, multiplier.shape={}".format(self.name, multiplier.shape))
            return False
        self.conv_weights = do_multiply2(self.fc_weights, multiplier)
        return True

    def add_bias(self, bias):
        # TODO: check the bias
        if self.have_bias:
            for i in range(0, bias.shape[0]):
                self.bias_weights[i] += bias[i]
        else:
            self.bias_weights = bias
            self.have_bias = True


class PoolNode(Node):
    def __init__(self, node):
        super(PoolNode, self).__init__(node)


class ExtractImagePatchesNode(Node):
    def __init__(self, node):
        super(ExtractImagePatchesNode, self).__init__(node)


class ResizeBilinearNode(Node):
    def __init__(self, node):
        super(ResizeBilinearNode, self).__init__(node)
        self.out_height = 0
        self.out_width = 0
        self.align_corners = ""


class AddNode(Node):
    def __init__(self, node):
        super(AddNode, self).__init__(node)
        self.can_merge = None
        self.container = []
        self.start_point = None


class ConcatNode(Node):
    def __init__(self, node):
        super(ConcatNode, self).__init__(node)
        self.container = []
        self.start_point = None
        self.axis = node.input[-1]


class MeanNode(Node):
    def __init__(self, node):
        # op = Mean
        super(MeanNode, self).__init__(node)
        self.kernel_h = 1
        self.kernel_w = 1
        self.stride_h = 1
        self.stride_w = 1


class BatchNormNode(Node):
    def __init__(self, node):
        # op = FusedBatchNorm
        super(BatchNormNode, self).__init__(node)
        self.mean = None
        self.variance = None
        self.e = None
        self.alpha = None
        self.beta = None

class ReluNode(Node):
    def __init__(self, node):
        super(ReluNode, self).__init__(node)
        self.alpha = None
        self.beta = None


cls_table = {
    "AvgPool": PoolNode,
    "MaxPool": PoolNode,
    "Conv2D": ConvNode,
    "DepthwiseConv2dNative": ConvNode,
    "ConcatV2": ConcatNode,
    "MatMul": FcNode,
    "Add": AddNode,
    "Mean": MeanNode,
    "FusedBatchNorm": BatchNormNode,
    "ReluNode": ReluNode,
    "ExtractImagePatches": ExtractImagePatchesNode,
    "Conv2DBackpropInput": DeConvNode,
}


class Model(object):
    def __init__(self, pb_path, input_node_names, output_node_names):
        self.pb_path = pb_path
        self.input_node_names = input_node_names
        self.output_node_names = output_node_names
        self.out_shape_dict = {}

        self.node_list = []
        self.name_node_dict = {}

        self.feed_dict = {}

        self.flag = None
        self.index_dict = defaultdict(int)

        self.struct = []
        self.weights = []

        # Record all the placeholders/node, each element is a Node
        self.place_holders = []
        # Record output node
        self.output_nodes = []

        self.first_fc = False
        self.alias_dict = {}

        self.init()

    def init(self):
        print("Get all the const values ... ")
        self.consts = get_consts(self.pb_path)
        print("Done.")

        with tf.gfile.GFile(self.pb_path, "rb") as g:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(g.read())
            _ = tf.import_graph_def(graph_def, name="")
            self.g_def = graph_def
            self.sess = tf.Session()

            # Get output node name if not defined
            if not self.output_node_names:
                _index = 0
                while 1:
                    _index -= 1
                    if self.g_def.node[_index].op in ["Reshape", "Shape", "Identity"]:
                        continue
                    self.output_node_names.append(self.g_def.node[_index].name)
                    break

            _input_shape = [0, 0, 0]
            switch_dict = {}
            remove_list = []
            for node in self.g_def.node:
                # Create node wrapper
                if node.op in cls_table:
                    _cls = cls_table[node.op]
                else:
                    _cls = Node
                _node = _cls(node)

                # Mark the const flag
                if node.name in self.consts:
                    _node.is_const = True
                else:
                    _node.is_const = False

                # Placeholder/input
                if (self.input_node_names and (node.name in self.input_node_names)) or \
                        (not self.input_node_names and node.op in ['Placeholder', 'Iterator', 'OneShotIterator']):
                    shape = self.get_tensor_shape(node.name).as_list()
                    shape = [x if x else -1 for x in shape]
                    _input_shape = [shape[1], shape[2], shape[3]]
                    if _input_shape[0] > 0 and  _input_shape[1] > 0 and _input_shape[2] > 0:
                        fake_data = np.ones(shape = (1, _input_shape[0], _input_shape[1], _input_shape[2]))
                    else:
                        fake_data = np.ones(shape = (1, 1, 1, 1))
                    self.feed_dict[self.get_tensor(node.name)] = fake_data
                    self.place_holders.append(_node)

                if node.op in ["Const", "Iterator", "OneShotIterator"]:
                    _node.attr = node.attr

                # Check data format
                elif node.op in ("Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput"):
                    _format = node.attr.get('data_format').s
                    if _format:
                        _node.data_format = decode_string(_format)

                # Record the output node
                if node.name in self.output_node_names:
                    self.output_nodes.append(_node)

                self.node_list.append(_node)
                self.name_node_dict[node.name] = _node

        # Check all the output nodes are found or not
        if len(self.output_nodes) != len(self.output_node_names):
            print("Cannot find all the output nodes")
            exit(-1)


    def optimize(self, extra_optimizer):
        # Make sure ONLY one placeholder
        if len(self.place_holders) > 1:
            print("More than one place holder detected, not support yet. Placeholders detected:")
            for holder in self.place_holders:
                print("\t%s" % holder.name)
            exit(-1)

        elif len(self.place_holders) < 1:
            print("Error: cannot detect the place holder.")
            exit(-1)

        # Create the graph (deduce input_nodes and output_nodes)
        for node in self.node_list:
            for _input in node.input:

                # Input may look like "Switch:1"
                m = re.search("(.+):[0-9]+", _input)
                if m:
                    _input = m.group(1)

                if _input.startswith('^'):
                    _input = _input[1:]

                # Record input node and output node
                input_node = self.name_node_dict.get(_input)
                if input_node:
                    node.input_nodes.append(self.name_node_dict.get(_input))
                    input_node.output_nodes.append(node)
                else:
                    logger.warning("Cannot find input node (%s) for %s." % (_input, node.name))

        # Detect circle
        visited = []
        visit_stack = []
        if self.contains_circle(self.place_holders[0], visited, visit_stack):
            print("Circle detected in the graph, currently cannot optimize it!");
            exit(-1)

        # Get all the nodes that the outputdepends on
        print("To get node dependencies ...")
        depends = self.get_depended_nodes(self.output_nodes)

        # Latest data format
        data_format = "NHWC"

        # Calc shape and weights
        print("To calculate shape and weights ...")
        for n in depends:
            n.output_shape = self.get_tensor_shape(n.name)
            if n.op in ("Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput"):
                n.conv_weights = self.get_weights(n)
                data_format = n.data_format
            elif n.op == "MatMul":
                n.fc_weights = self.get_weights(n)
            elif n.op == "FusedBatchNorm":
                n.e = n._node.attr["epsilon"].f
                _scale, _beta, _mean, _variance = [
                    self.consts.get(x) for x in n.input[1:]]
                if _scale is not None:
                    _len = len(_scale)
                    if not isinstance(_mean, np.ndarray): _mean = np.zeros(_len)
                    if not isinstance(_variance, np.ndarray): _variance = np.ones(_len)
                    n.mean = _mean
                    n.variance = _variance
                    n.alpha = _scale
                    n.beta = _beta
            elif n.op == "Mean":
                _input = self.get_input_names(n)[0]
                _shape = self.get_tensor_shape(_input)
                if data_format == 'NHWC':
                    n.kernel_h, n.kernel_w = _shape[1], _shape[2]
                else:
                    n.kernel_h, n.kernel_w = _shape[2], _shape[3]
            elif n.op == "ResizeBilinear":
                _shape = self.get_weights(n)
                if _shape:
                    n.out_height = _shape[0]
                    n.out_width = _shape[1]
                else:
                    logger.warning("Cannot get output height and width for node: %s" % n.name)
                    n.out_height = -1
                    n.out_width = -1
                n.align_corners = str(n._node.attr["align_corners"].b)
                    

        # Detect 'Merge', 'Switch' nodes, remove it if possible
        print("To remove unnecessary nodes ...")
        self.erase_identity_node(depends)
        self.erase_merge_node(depends)
        self.erase_switch_node(depends)
        self.erase_reshape_node(depends)

        # Only deal with the needed node (dependency changed, so get depended node again)
        depends = self.get_depended_nodes(self.output_nodes)

        # Merge some nodes if applicable
        while True:
            merged_node_num = 0
            merged_node_num += self.merge_add_mul(depends)
            merged_node_num += self.merge_batchnorm(depends)
            merged_node_num += self.merge_pad_node(depends)
            merged_node_num += self.merge_mul_maximum(depends)

            # Cannot merge it any more, then break
            if merged_node_num == 0:
                break
            else:
                depends = self.del_unused_nodes(depends)

        # Deduce the node calculation order
        logger.info("To deduce output node sequence ...")
        self.deduce_calc_seq(depends)

        # Check the calculation sequence for output node whether ready
        for output_node in self.output_nodes:
            if not output_node.calc_seq:
                print("Cannot get the calc sequence for node: %s" % output_node.name)
                exit(-1)

    # Check if the graph contains circle
    def contains_circle(self, node, visited, visit_stack):
        # Use DFS to detect circle, thus we define a stack
        visited.append(node)
        visit_stack.append(node)

        for child in node.output_nodes:
            # visited node does not need to check it again (prune)
            if child not in visited:
                if self.contains_circle(child, visited, visit_stack):
                    return True
            elif child in visit_stack:
                return True

        # Remember to pop the node
        visit_stack.pop()
        return False


    # Try to erase the 'Identity' node
    def erase_identity_node(self, node_list):
        for _node in node_list:
            if _node.op != 'Identity':
                continue

            if len(_node.input_nodes) == 1:
                _node.merged = True
                merged_to = _node.input_nodes[0]
                self.fix_graph(_node, merged_to)

    # Get all the names of a node list
    def get_node_names(self, nodes):
        names = ""
        for _node in nodes:
            names = names + "," + _node.name
        return names

    # Merge the pattern like: A(Conv2D) -> B(Mul) -> C(Maximum), A -> C
    def merge_mul_maximum(self, node_list):
        merged_nodes = 0
        for _node in node_list:
            if _node.merged:
                continue

            if _node.op != 'Maximum':
                continue

            if len(_node.input_nodes) != 2:
                logger.info("Input of Maximum({}) is not 2".format(_node.name))
                continue

            # Get mul node and the other node
            a = None
            b = None
            c = _node
            for _input in _node.input_nodes:
                if _input.op == 'Mul':
                    b = _input
                else:
                    a = _input

            if a is None or b is None:
                continue

            if len(b.input_nodes) != 2:
                logger.info("Input of Mul({}) is not 2".format(b.name))
                continue

            if a not in b.input_nodes:
                logger.info("Maximum({}) does not match the pattern to merge".format(c.name))
                continue

            # At this point, could replace B and C with a Relu node
            relu_node = ReluNode(c._node)
            relu_node.op = 'Relu'
            relu_node.output_shape = c.output_shape
            relu_node.alpha = self.get_weights(b)

            del a.output_nodes[a.output_nodes.index(b)]
            del a.output_nodes[a.output_nodes.index(c)]
            a.output_nodes.append(relu_node)
            relu_node.output_nodes.extend(c.output_nodes)
            relu_node.input_nodes.append(a)
            for _output in c.output_nodes:
                _output.input_nodes[_output.input_nodes.index(c)] = relu_node

            c.merged = True
            b.merged = True
            merged_nodes += 1

        return merged_nodes


    # Try to eease the 'Pad' node
    def merge_pad_node(self, node_list):
        merged_nodes = 0
        for _node in node_list:
            if _node.merged:
                continue

            if _node.op != 'Pad':
                continue

            inputs = self.get_nonconst_input(_node)
            if len(inputs) != 1:
                logger.info("Pad({}) has more than 1 input: {}".format(_node.name, self.get_node_names(inputs)))
                continue

            if len(_node.output_nodes) != 1:
                logger.info("Pad({}) has more than 1 output: {}".format(_node.name, self.get_node_names(_node.output_nodes)))
                continue

            # Erase the pad node only when the following node is 'Conv2D'
            node_before = inputs[0]
            node_after = _node.output_nodes[0]
            if node_after.op not in ["Conv2D", "Conv2DBackpropInput"]:
                logger.info("Pad({}) is followed by {}({})".format(_node.name, node_after.op, node_after.name))
                continue

            # Add the padding value to 'Conv2D' node
            l_pad, r_pad, t_pad, b_pad = self.parse_pad(_node, node_after.data_format)
            _node.pad = [l_pad, r_pad, t_pad, b_pad]
            node_after.pad += _node.pad

            # Fix the graph to skip the pad node
            node_before.output_nodes[node_before.output_nodes.index(_node)] = node_after
            node_after.input_nodes[node_after.input_nodes.index(_node)] = node_before
            _node.merged = True

            merged_nodes += 1
            logger.info("Pad({}) is merged".format(_node.name))
        return merged_nodes


    # Return true if the node is invalid (for example: one input of merge node)
    def is_node_useless(self, node_name):
        try:
            self.run_sess(node_name)
            return False
        except Exception as e:
            return True
        

    # Try to erase the 'Merge' node
    def erase_merge_node(self, node_list):
        for _node in node_list:
            if _node.op != "Merge":
                continue

            # To check whether can get the value, if cannot, then useless
            if len(_node.input_nodes) == 2:
                first_node = _node.input_nodes[0]
                second_node = _node.input_nodes[1]
                # Not validate the value yet
                if first_node.is_useless is None or second_node.is_useless is None:
                    first_node.is_useless = self.is_node_useless(first_node.name)
                    second_node.is_useless = not first_node.is_useless
                    print("Merge input - " + first_node.name + (" is invalid" if first_node.is_useless else " is valid"))
                    print("Merge input - " + second_node.name + (" is invalid" if second_node.is_useless else " is valid"))
            else:
                for _input in _node.input_nodes:
                    _input.is_useless = is_node_useless(_input)

            # Remove useless/invalid input
            trimmed_input = []
            for _input in _node.input_nodes:
                if not _input.is_useless:
                    trimmed_input.append(_input)
            _node.input_nodes = trimmed_input

            # Only one valid input
            if len(trimmed_input) == 1:
                _node.merged = True
                self.fix_graph(_node, trimmed_input[0])


    # Erase 'Switch' node
    def erase_switch_node(self, node_list):
        for _node in node_list:
            if _node.op != "Switch":
                continue

            nonconst_inputs = []
            for _input in _node.input_nodes:
                if not _input.is_const: nonconst_inputs.append(_input)

            if len(nonconst_inputs) == 1:
                _node.merged = True
                self.fix_graph(_node, nonconst_inputs[0])


    # Erase 'Reshape' node
    def erase_reshape_node(self, node_list):
        for _node in node_list:
            if _node.op != "Reshape":
                continue

            nonconst_inputs = []
            for _input in _node.input_nodes:
                if _input.op != "Pack": nonconst_inputs.append(_input)

            _node.merged = True
            self.fix_graph(_node, nonconst_inputs[0])

    # Deduce/speculate the node calculation sequence
    # Start from placeholder, and then use a algorithm like Dynamic Programming
    def deduce_calc_seq(self, node_list):
        start_node = self.place_holders[0]
        start_node.calc_seq.append(start_node)

        idx = 0
        work_queue = [start_node]
        while idx < len(work_queue):
            # Get a node from the queue
            cur_node = work_queue[idx]
            #print("node: %s, depth: %d" % (cur_node.name, len(cur_node.calc_seq)))
            # Try to decide the calc seq of its outputs
            for _node in cur_node.output_nodes:
                # The node is already visited
                if _node in work_queue:
                    #print("\t%s already in the queue" % _node.name)
                    continue
                if self.determine_calc_seq(_node):
                    work_queue.append(_node)
                    #print("\tChild node %s added to queue" % _node.name)
                else:
                    pass
                    #print("\tCannot decide calc seq for %s" % _node.name)
            idx += 1


    # Dump the calc sequence
    def dump_calc_seq(self, node):
        print("Calc sequence of %s" % node.name)
        for _node in node.calc_seq:
            print("\t%s (%s)" % (_node.name, _node.op))


    # Try to determine calc sequence for a node
    # Return True if succeed, otherwise return False
    def determine_calc_seq(self, node):
        seq = []
        for _node in node.input_nodes:
            # Ignore const input
            if _node.is_const:
                continue
            # Calc seq of input node is not determinated yet
            if not _node.calc_seq:
                return False
            # The first non-const input
            elif not seq:
                seq.extend(_node.calc_seq)
            # For Add/Concat node, it will reach here
            else:
                seq = self.merge_calc_seq(seq, _node.calc_seq)

        seq.append(node)
        node.calc_seq = seq

        return True


    # Merge 2 sequences, make sure following the seqeunce can get the value
    def merge_calc_seq(self, seq1, seq2):
        for _node in seq2:
            if _node not in seq1:
                seq1.append(_node)
        return seq1


    # Delete all the merged/useless nodes, return the new list
    def del_unused_nodes(self, node_list):
        result = []
        for _node in node_list:
            if _node.merged or _node.is_useless:
                continue
            result.append(_node)
        return result


    # Merge the batch norm to previous Conv
    def merge_batchnorm(self, node_list):
        merged_nodes = 0

        for n in node_list:
            if n.merged:
                continue

            if n.op == "FusedBatchNorm":
                nonconst_inputs = []
                for _input in n.input_nodes:
                    if not _input.is_const: nonconst_inputs.append(_input)

                # The only input is Convolution
                if len(nonconst_inputs) == 1 and \
                    (nonconst_inputs[0].op in ("Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput")):
                    pre_node = nonconst_inputs[0]
                    _mean = n.mean
                    _variance = n.variance
                    _epsilon = n.e
                    _scale = n.alpha
                    _beta = n.beta
                    if not pre_node.have_bias:
                        if pre_node.data_format == 'NHWC':
                            _bias = np.zeros(pre_node.conv_weights.shape[-1])
                        else:
                            _bias = np.zeros(pre_node.conv_weights.shape[1])
                    else:
                        _bias = pre_node.bias_weights
                    try:
                        _alpha = _scale / np.sqrt(_variance + _epsilon)
                    except Exception as e:
                        print('ERROR: Fail to cope FusedBatchNorm[%s]' % n.name)
                        print('scale:' +  _scale)
                        print('variance:' +  _variance)
                        print('epsilon:' +  _epsilon)
                        exit()
                    pre_node.bias_weights = _beta - _alpha * _mean
                    pre_node.have_bias = True
                    pre_node.conv_weights = do_multiply(
                        pre_node, pre_node.conv_weights, _alpha)
                    n.merged = True
                    merged_nodes += 1

                    # Fix the graph (by changing input_nodes and output_nodes)
                    self.fix_graph(n, pre_node)

        return merged_nodes


    # Merge Add/BiasAdd to Conv/MatMul
    # Merge Mul to Convolution
    def merge_add_mul(self, node_list):
        merged_nodes = 0

        for n in node_list:
            if n.merged:
                continue

            if n.op == "BiasAdd" or n.op == "Add" or n.op == "Mul":
                # Get const input and non-const input node
                const_inputs = []
                nonconst_inputs = []
                for _input in n.input_nodes:
                    if _input.is_const: const_inputs.append(_input)
                    else: nonconst_inputs.append(_input)

                # input node is Conv2D/Matmul
                if len(const_inputs) == 1 and len(nonconst_inputs) == 1 and \
                    (nonconst_inputs[0].op in ("MatMul", "Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput")):

                    pre_node = nonconst_inputs[0]
                    merged = None
                    if n.op == "Mul":
                        multiplier = self.get_weights(n)
                        merged = pre_node.mul(multiplier)
                    else:
                        pre_node.have_bias = True
                        tmp_bias_weights = self.get_weights(n)
                        if not isinstance(pre_node.bias_weights, type(None)):
                            tmp_bias_weights = tmp_bias_weights + pre_node.bias_weights
                        pre_node.bias_weights = tmp_bias_weights
                        merged = True

                    if merged:
                        n.merged = True
                        merged_nodes += 1

                        # Fix the graph (by changing input_nodes and output_nodes)
                        self.fix_graph(n, pre_node)

                        logger.info("Merged {} to {}".format(n.name, pre_node.name))

        return merged_nodes

    # After a node is merged, fix the graph (by changing input_nodes and output_nodes)
    # merged_node: which node is merged
    # merged_to: the node where is merged to
    def fix_graph(self, merged_node, merged_to):
        for _node in merged_node.input_nodes:
            if _node.is_const:
                # The const node is ONLY used for merged node, then can be removed
                if len(_node.output_nodes) == 1:
                    _node.merged = True
                continue
            # Add merged node's output to preceding nodes' output
            del _node.output_nodes[_node.output_nodes.index(merged_node)]
            _node.output_nodes.extend(merged_node.output_nodes)

        # Modify the input of (the output node of merged node)
        for _node in merged_node.output_nodes:
            _node.input_nodes[_node.input_nodes.index(merged_node)] = merged_to

        # Revise the output to make sure it can be found
        if merged_node in self.output_nodes:
            self.output_nodes[self.output_nodes.index(merged_node)] = merged_to

    # Get all the nodes that the target_node depends on
    def get_depended_nodes(self, target_nodes):
        result = list(target_nodes)

        # Enumerate all the dependencies in 'result' list
        idx = 0
        while idx < len(result):
            # the node is an input/placeholder
            if result[idx] in self.place_holders:
                idx += 1
                continue
            for dep in result[idx].input_nodes:
                # the node is not added to the list yet
                if dep not in result:
                    result.append(dep)
            idx += 1

        return result

    # To evaluate a tensor value
    def run_sess(self, node_name):
        return self.sess.run('%s:0' % node_name, feed_dict = self.feed_dict)

    def get_sub_nodes(self, node_name):
        sub_nodes = []
        n = self.name_node_dict.get(node_name)
        # struct input tree, short input tree, wirte to topo list
        ns = tf.graph_util.extract_sub_graph(self.g_def, [n.name])
        for n in ns.node:
            #if n.op in ['Const', 'Identity']: continue
            sub_nodes.append(n.name)

        return sub_nodes

    # Find node by name
    def find_node(self, node_name):
        for _node in self.node_list:
            if _node.name == node_name:
                return _node
        return None


    # Parse padding values for 'Pad' node
    def parse_pad(self, node, data_format):
        for ip in node.input_nodes:
            if ip.op == 'Const':
                dim = ip.attr['value'].tensor.tensor_shape.dim
                row, col = int(dim[0].size), int(dim[1].size)
                tensor_content = str(ip.attr['value'].tensor).split('tensor_content')[1]
                tensor_content = tensor_content.split(': "\\')[1].strip('\n').split('\\')
                num = 0
                if data_format == 'NCHW': num = 2
                l_pad, r_pad, t_pad, b_pad = int(tensor_content[row*(col+num)]), \
                                    int(tensor_content[row * (col+num+1)]), \
                                    int(tensor_content[row * (col+num+2)]), \
                                    int(tensor_content[row * (col+num+3)])
                #print 'PAD:', node.name, l_pad, r_pad, t_pad, b_pad
        return l_pad, r_pad, t_pad, b_pad

    def get_weights(self, node):
        const_input_nodes = []
        for n in node.input_nodes:
            if n.is_const:
                const_input_nodes.append(n)

        # Only 1 const input
        if len(const_input_nodes) == 1:
            return self.consts.get(const_input_nodes[0].name)

        # More than 1 const input
        elif len(const_input_nodes) > 1:
            for n in const_input_nodes:
                if n.op in ["Identity", "Const"]:
                    return self.consts.get(n.name)


    # Get non-const input for a node
    def get_nonconst_input(self, node):
        nonconst_inputs = []
        for _input in node.input_nodes:
            if not _input.is_const: 
                nonconst_inputs.append(_input)
        return nonconst_inputs


    def get_tensor(self, node_name):
        return self.sess.graph.get_tensor_by_name(node_name + ":0")

    def get_tensor_shape(self, node_name):
        return self.sess.graph.get_tensor_by_name(node_name + ":0").shape

    def get_input_shape(self, node):
        for _input in node.input_nodes:
            if not _input.is_const:
                return _input.output_shape
        return None


    # Get input nodes which excluded const nodes
    # Node: the graph may be merged, thus cannot get from _node.input
    def get_input_names(self, node):
        input_names = []
        for _input in node.input_nodes:
            if not _input.is_const:
                input_names.append(_input.name)
        return input_names


    # Return input channel, height, width
    def get_graph_input(self):
        input_node = self.place_holders[0]
        input_shape = input_node.output_shape

        # Default data format
        data_format = "NHWC"

        # Traverse the output to find a conv node to get the data format
        work_queue = [input_node]
        idx = 0
        while idx < len(work_queue):
            cur_node = work_queue[idx]
            if cur_node.op in ("Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput"):
                data_format = cur_node.data_format
                break
            for _node in cur_node.output_nodes:
                work_queue.append(_node)
            idx += 1

        shape = input_shape.as_list()
        shape = [(x if x else -1) for x in shape]

        # Return value in the order of channel, height, width
        if data_format == "NHWC":
            return shape[3], shape[1], shape[2]
        elif data_format == "NCHW":
            return shape[1], shape[2], shape[3]
        else:
            return None


    # Get input tensor format, like 'NCHW', 'NHWC', 'HW'
    def get_input_format(self, node):
        _inputs = self.get_nonconst_input(node)

        # Loop back to deeper input to get data format
        while len(_inputs) > 0:
            n = _inputs[0]
            if n.op in ("Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput"):
                return n.data_format
            elif n.op == "MatMul":
                return "HW"
            _inputs = self.get_nonconst_input(n)


    def _get_nChwxc(self):
        content = ''
        x_of_nChwxc = 8
        with open('/proc/cpuinfo', 'r') as f:
            content = f.read()
        for line in content.split('\n'):
            if len(line) == 0: continue
            if line[:5] != 'flags': continue
            if 'avx512f' in line: x_of_nChwxc = 16
            break

        return x_of_nChwxc

    # Write graph and weights to file
    def write_to_file(self, topo_file, weights_file):
        x_of_nChwxc = self._get_nChwxc()
        dumped_nodes = []

        for output_node in self.output_nodes:
            for n in output_node.calc_seq:
                if n in dumped_nodes:
                    continue
                if self.first_fc:
                    #   In the reason that it's necessary to reshape the input weights of fc layer from nchw to nChw[x]c of mkldnn,
                    # so we statistic output shape of common layers.
                    if n.op in ("Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput", "AvgPool", "MaxPool", "Mean", "ExtractImagePatches"):
                        fake_out = self.sess.run('%s:0'% n.name, feed_dict = self.feed_dict)
                        self.out_shape_dict[n.name] = fake_out.shape
    
                logger.info("Dump node, op:%s, name: %s" %(n.op, n.name))
                if n.op in ("ConcatV2", "Add", "Softmax"):
                    dump_simple_op(n._node, n.name, self.get_input_names(n), topo_file)
                elif n.op == "Placeholder" \
                        or n.op == "Iterator" \
                        or n.op == "OneShotIterator":
                    channel, height, width = self.get_graph_input()
                    dump_placeholder(n.name, topo_file, height, width, channel)
                elif n.op in ("AvgPool", "MaxPool"):
                    input_shape = self.get_input_shape(n)
                    dump_pool(n._node, n.name, self.get_input_names(n),
                              input_shape, n.output_shape, topo_file)
                elif n.op in ("Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput"):
                    input_shape = self.get_input_shape(n)
                    dump_convolution(n._node, n.name, self.get_input_names(n), n.conv_weights, n.have_bias,
                                     n.bias_weights, input_shape, n.output_shape, n.pad, topo_file, weights_file)
                elif n.op == "MatMul":
                    input_shape = self.get_input_shape(n)
                    input_format = self.get_input_format(n)
                    dump_fc(n._node, n.name, self.get_input_names(n), self.first_fc, input_shape, n.fc_weights,
                            n.have_bias, n.bias_weights, n.output_shape[1], topo_file, weights_file, input_format, x_of_nChwxc)
                    self.first_fc = False
                elif n.op == "Mean":
                    dump_mean(n._node, n.name, self.get_input_names(n), n.kernel_h, n.kernel_w, n.stride_h, n.stride_w, topo_file)
                elif n.op == "FusedBatchNorm":
                    dump_batchnorm(n._node, n.name, self.get_input_names(n), n.mean, n.variance, n.e, n.alpha, n.beta, topo_file, weights_file);
                elif n.op in ("Relu", "Relu6"):
                    dump_relu(n, n.name, self.get_input_names(n), topo_file)
                # For debug usage
                elif n.op == "Mul" or n.op == "Maximum":
                    dump_simple_op(n._node, n.name, self.get_input_names(n), topo_file)
                elif n.op == "ExtractImagePatches":
                    input_shape = self.get_input_shape(n)
                    dump_extract_image_patches(n._node, n.name, self.get_input_names(n),
                              input_shape, n.output_shape, topo_file)
                elif n.op == "ResizeBilinear":
                    dump_resize_bilinear(n._node, n.name, n.out_height, n.out_width, n.align_corners, self.get_input_names(n), topo_file)
                else:
                    # Ignore the node by fixing input/output around the node, so that no node will use it as the input
                    nonconst_inputs = self.get_nonconst_input(n)
                    if len(nonconst_inputs) == 1:
                        self.fix_graph(n, nonconst_inputs[0])
                        logger.warning("Ignored %s, %s" % (n.op, n.name))
                    else:
                        logger.error("Cannot handle %s, %s" % (n.op, n.name))
                        for _node in nonconst_inputs:
                            print("  ", _node)
                        #exit(-1)

                dumped_nodes.append(n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_model_filename", default="./mobilenet_224.pb",
                        type=str, help="Frozen model file to read")
    parser.add_argument("--input_node_name", default="",
                        type=str, help="Input node name, default is the Placeholder")
    parser.add_argument("--output_node_name", default="",
                        type=str, help="The last node where to get prediction result. "
                                       "If there are multiple output nodes, separate them with ','")
    parser.add_argument("--weights_file", default="./weights.bin",
                        type=str, help="File to dump the weights")
    parser.add_argument("--pkl_file", default="./weights.pkl",
                        type=str, help="File to dump the pickle format weights")
    parser.add_argument("--topo_file", default="./topo.txt",
                        type=str, help="File to dump the topology")

    args = parser.parse_args()

    in_names = args.input_node_name.split(',') if args.input_node_name else [];
    out_names = args.output_node_name.split(',') if args.output_node_name else [];
    m = Model(args.input_model_filename, in_names, out_names)

    m.optimize(None)

    print("\nBegin to dumping ...")
    with open(args.weights_file, "wb") as weights_file:
        with open(args.topo_file, "wb") as topo_file:
            m.write_to_file(topo_file, weights_file)

    # Dump to packle (needed when generate caffe model)
    with open(args.pkl_file, "wb") as f:
        pickle.dump(g_weights, f)

    logger.info("Convert Done.")
