#!/usr/bin/env python

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework.attr_value_pb2 import AttrValue
from tensorflow.core.protobuf.saved_model_pb2 import SavedModel
import tensorflow as tf

from itertools import groupby
import logging
import argparse
import os
import sys
from typing import List, Tuple, Any
from types import ModuleType

from model_modifier import LOGGER_FORMAT, BERT_OP_ENV_VAR, TENSORS_PER_LAYER
from model_modifier.bert_op_helper import BertOpHelper

log = logging.getLogger(f'{os.path.basename(__file__)}')

def flatten_nodes(graphs: List[GraphDef]) -> List[Tuple[NodeDef, str]]:
    # Flatten all nodes into a single iterable.
    # This means all nodes in all graphs, and in all function defs of each graph.
    return [(node, str('graph_def')) for graph in graphs for node in graph.node] + \
        [(node, str(function.signature.name)) for graph in graphs for function in graph.library.function for node in function.node_def]


def filter_by_op(nodes: List[Tuple[NodeDef, str]], op: str) -> List[Tuple[NodeDef, str]]:
    return [node for node in nodes if node[0].op == op]


def configure_bert_op_nodes(graphs: List[GraphDef], args: argparse.Namespace) -> None:

    def _all_attrs_same(nodes: List[NodeDef]) -> bool:
        attrs = [node.attr for node in nodes]
        g = groupby(attrs)
        return next(g, True) and not next(g, False)

    def _log_node_config(node: NodeDef, log_func) -> None:
        log_func('Bert op configuration:')
        sorted_attrs = sorted(node.attr.items())
        for key, value in sorted_attrs:
            v = ' '.join(str(value).split()) # Collapse multiline attrs into a single line
            log_func(f'  {key}: {v}')

    def _reset_attributes(node: NodeDef) -> None:
        av = AttrValue()
        av.CopyFrom(node.attr['_output_shapes'])
        node.attr.clear()
        node.attr['_output_shapes'].CopyFrom(av)

    def _configure_attributes(args: argparse.Namespace, node: NodeDef) -> None:
        if args.mask_type is not None:
            node.attr['MaskT'].type = args.mask_type.as_datatype_enum
        if args.quantization is not None:
            node.attr['QuantizableDataType'].type = tf.qint8.as_datatype_enum if args.quantization else tf.float32.as_datatype_enum
            # node.attr['UseQuantization'].type = tf.qint8.as_datatype_enum if args.quantization else tf.float.as_datatype_enum
        if args.bfloat16 is not None:
            node.attr['NonQuantizableDataType'].type = tf.bfloat16.as_datatype_enum if args.bfloat16 else tf.float32.as_datatype_enum
            # node.attr['UseBFloat16'].type = tf.bfloat16.as_datatype_enum if args.bfloat16 else tf.float.as_datatype_enum
        if args.layers is not None:
            node.attr['NumWeights'].i = TENSORS_PER_LAYER * args.layers
        if args.hidden_size is not None:
            node.attr['HiddenSize'].i = args.hidden_size
        if args.num_attention_heads is not None:
            node.attr['NumAttentionHeads'].i = args.num_attention_heads
        if args.intermediate_size is not None:
            node.attr['IntermediateSize'].i = args.intermediate_size
        if args.max_seq_len is not None:
            node.attr['MaxSequenceLength'].i = args.max_seq_len
        if args.activation is not None:
            node.attr['HiddenAct'].s = args.activation.encode()
        if args.calibrate_quant_factors is not None:
            node.attr['CalibrateQuantFactors'].b = args.calibrate_quant_factors
        if args.quantization_factors_path is not None:
            node.attr['QuantizationFactorsPath'].s = args.quantization_factors_path.encode()

    def _validate_bert_node(bert_module: ModuleType, node: NodeDef) -> None:
        if node.op != 'Bert':
            raise ValueError(f'Node {node.name} must be a \'Bert\' op. It is {node.op}.')
        
        b = BertOpHelper(lib=bert_module,
            mask_type=tf.dtypes.as_dtype(node.attr['MaskT'].type),
            max_token_size=node.attr['MaxSequenceLength'].i,
            num_weights=node.attr['NumWeights'].i,
            hidden_size=node.attr['HiddenSize'].i,
            num_attention_heads=node.attr['NumAttentionHeads'].i,
            intermediate_size=node.attr['IntermediateSize'].i,
            quantizable_datatype=tf.dtypes.as_dtype(node.attr['QuantizableDataType'].type),
            non_quantizable_datatype=tf.dtypes.as_dtype(node.attr['NonQuantizableDataType'].type),
            hidden_act=node.attr['HiddenAct'].s.decode(),
            calibrate_quant_factors=node.attr['CalibrateQuantFactors'].b,
            quantization_factors_path=node.attr['QuantizationFactorsPath'].s.decode()
        )
        try:
            b.call()
        except Exception as e:
            raise RuntimeError(f'Attribute validation failed: {e}')

    nodes_with_function_name = flatten_nodes(graphs)
    bert_nodes_with_function_name = filter_by_op(nodes_with_function_name, 'Bert')

    if len(bert_nodes_with_function_name) == 0:
        raise RuntimeError('No Bert op nodes were located in the model.')

    log.info('The following BERT nodes will be updated:')
    for node, func in bert_nodes_with_function_name:
        log.info(f'  Node \'{node.name}\' in \'{func}\'')

    # Drop the function names, we do not need them after this point
    bert_nodes = [node[0] for node in bert_nodes_with_function_name]

    if args.reset:
        for node in bert_nodes:
            _reset_attributes(node)

    for node in bert_nodes:
        _configure_attributes(args, node)

    bert_nodes[0].attr
    if not _all_attrs_same(bert_nodes):
        raise RuntimeError('Not all Bert nodes in the model have the same attribute values. You can fix this by using the '
                  '--reset flag and setting all attribute values again.')
    else:
        log.debug('Attributes of all Bert nodes are identical.')

    # At this point we know there is at least one Bert node in the model and that all Bert nodes have identical
    # attribute values, so we can just print and validate the first one:
    _log_node_config(bert_nodes[0], log.info)

    if args.bert_module is not None:
        _validate_bert_node(args.bert_module, bert_nodes[0])
        log.info('Attribute validation was successful.')
    else:
        log.warn('Attribute validation was NOT performed.')



class TFLoadOpLibraryAction(argparse.Action):

    def __init__(self, *args: Any, **kwargs: Any):
        super(TFLoadOpLibraryAction, self).__init__(*args, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None) -> None:
        try:
            if isinstance(values, list):
                if len(values) != 1:
                    raise argparse.ArgumentError(self, f'Exactly one argument value is required.')
                value = values[0]
            else:
                value = values

            module = TFLoadOpLibraryAction.load_and_fail_if_no_bert_wrapper(value)

            setattr(namespace, self.dest, module)

        except (RuntimeError, tf.errors.NotFoundError) as e:
            raise argparse.ArgumentError(self, f'Failed to load op library. what(): {e}')
    
    @staticmethod
    def load_from_env(var_name: str) -> ModuleType:
        path = os.getenv(var_name)

        if path is None:
            return None

        # Make sure the loaded .so contains a Bert wrapper.
        # You can actually call `tf.load_op_library('')` and it will load something, obviously not the BertOp library.
        module = TFLoadOpLibraryAction.load_and_fail_if_no_bert_wrapper(path)
        return module


    @staticmethod
    def load_and_fail_if_no_bert_wrapper(path: str) -> ModuleType:
        module = tf.load_op_library(path)
        if not hasattr(module, 'Bert'):
                raise RuntimeError(f'Shared object library \'{path}\' does not contain a Bert wrapper.')
        return module


class TFDataTypeAction(argparse.Action):
    def __init__(self, *args: Any, **kwargs: Any):
        super(TFDataTypeAction, self).__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        def str_to_tf_dtype(dtype_as_str: str) -> tf.DType:
            dtype = getattr(tf, dtype_as_str)
            if not isinstance(dtype, tf.DType):
                raise ValueError()
            return dtype

        if isinstance(values, list):
            if len(values) != 1:
                raise argparse.ArgumentError(self, f'Exactly one argument value is required.')
            value = values[0]
        else:
            value = values

        try:
            dtype = str_to_tf_dtype(value)
            setattr(namespace, self.dest, dtype)
        except (AttributeError, ValueError):
            raise argparse.ArgumentError(self, f'{value} is not a tensorflow.DType')




def main():
    parser = argparse.ArgumentParser(
        description='Configure attributes of BertOp nodes in a model.', add_help=False)

    parser.add_argument('--help', action='help')

    parser.add_argument(
        'model', help='Saved model folder or a frozen model .pb file')
    
    parser.add_argument('--bert-op-lib', dest='bert_module', action=TFLoadOpLibraryAction, default=None,
                        help=f'Path to the BertOp library .so. If not provided, it will be read from the '
                        '{BERT_OP_ENV_VAR} environment variable. If that is not set, the program will log a warning '
                        'and continue WITHOUT validating the configuration.')

    parser.add_argument('-r', '--reset', dest='reset', action='store_true', default=False,
                        help='Reset the BertOp nodes by clearing all attribute values except _output_shapes. '
                        'This is done before any other attributes are applied.')

    parser.add_argument('--log', dest='log_level', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL', 'FATAL'],
                        default='INFO', help='Logging verbosity level.')

    parser.add_argument('-m', '--mask-type', dest='mask_type', action=TFDataTypeAction, default=None,
                        help='Expected data type of the mask tensor.')

    parser.add_argument('-q', '--quantization', dest='quantization', action='store_true', default=None,
                        help='Use int8 quantization.')
    parser.add_argument('-Q', '--no-quantization', dest='quantization', action='store_false',
                        help='Do not use int8 quantization.')

    parser.add_argument('-b', '--bfloat16', dest='bfloat16', action='store_true', default=None,
                        help='Use BFloat16 in supported operations.')
    parser.add_argument('-B', '--no-bfloat16', dest='bfloat16', action='store_false',
                        help='Do not use BFloat16 in supported operations.')

    parser.add_argument('-l', '--layers', dest='layers', type=int, default=None,
                        help='Number of layers of the BERT model.')

    parser.add_argument('-h', '--hidden-size', type=int, default=None,
                        help='Hidden size of the model.')

    parser.add_argument('-a', '--num-attention-heads', type=int, default=None,
                        help='Number of attention heads of the model.')

    parser.add_argument('-i', '--intermediate-size', type=int, default=None,
                        help='Intermediate size of the model.')

    parser.add_argument('-s', '--max-seq-len', type=int, default=None,
                        help='Max length of the token input sequence.')

    parser.add_argument('-A', '--activation', type=str, default=None, choices=['gelu_tanh', 'gelu_erf'],
                        help='Type of activation function to be used by the BertOp. This attribute is currently '
                        'ignored.')

    parser.add_argument('-c', '--calibrate', dest='calibrate_quant_factors', action='store_true', default=None,
                        help='Enable calibration mode to determine INT8 quantization factors. '
                        'This option can only be used in pure FP32 mode, i.e. --no-bfloat16 --no-quantization.')
    parser.add_argument('-C', '--no-calibrate', dest='calibrate_quant_factors', action='store_false',
                        help='Disable calibration mode.')

    parser.add_argument('-p', '--quant-factors-path', dest='quantization_factors_path', default=None, type=str,
                        help='Path to save/load the quantization factors file to/from. '
                        'Ignored in float mode, unless --calibrate is used.')


    parser.add_argument('-o', '--output', default=None,
                        help='Location of the output .pb. If not provided, the model will be modified in-place.')


    args = parser.parse_args()


    logging.basicConfig(format=LOGGER_FORMAT)
    logging.root.setLevel(args.log_level)


    if args.bert_module is None:
        log.info(f'BertOp library path was not provided. Attempting to load it from the {BERT_OP_ENV_VAR} environment '
                 'variable.')
        try:
            module = TFLoadOpLibraryAction.load_from_env(BERT_OP_ENV_VAR)
        except (RuntimeError, tf.errors.NotFoundError) as e:
            raise RuntimeError(f'Failed to load the BertOp library from \'{BERT_OP_ENV_VAR}\'. Clear the variable or '
                               f'make sure it points to the correct location. Original error: {e}')
        if module is not None:
            log.info(f'Library loaded from {BERT_OP_ENV_VAR} environment variable.')
            args.bert_module = module
        else:
            log.warn('BertOp library was NOT loaded. The script will continue WITHOUT validation '
                         'of the attribute values.')

    if os.path.isdir(args.model):
        saved_model_path = os.path.join(args.model, 'saved_model.pb')
        if not args.output:
            args.output = saved_model_path

        if tf.saved_model.contains_saved_model(args.model):
            print('Provided directory potentially contains a saved model. Attempting to load the saved model graph...')
            saved_model = SavedModel()
            with open(saved_model_path, 'rb') as f:
                saved_model.ParseFromString(f.read())
            graphs = [
                meta_graph.graph_def for meta_graph in saved_model.meta_graphs]
            print('Saved model graph loaded successfully.')
        else:
            raise ValueError(
                'Provided directory does not contain a saved model.')
        output_proto = saved_model
    else:
        if not args.output:
            args.output = args.model

        graph = GraphDef()
        print('Attempting to load the frozen model graph...')
        with open(args.model, 'rb') as f:
            graph.ParseFromString(f.read())
        print('Frozen model graph loaded successfully.')
        graphs = [graph]
        output_proto = graph

    configure_bert_op_nodes(graphs, args)

    with open(args.output, 'wb') as f:
        f.write(output_proto.SerializeToString())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f'Unexpected error: {e}')
        log.warn(f'No changes were made to the model.')
        exit(1)
