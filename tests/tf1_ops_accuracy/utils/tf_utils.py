# Copyright 2019 The TF-STK Authors. All Rights Reserved.
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
"""Utility Functions"""

from functools import wraps
import os
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat

from . import stk_logging
logger = stk_logging.get_logger()


def suppress_tf_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save log settings
        verbosity = tf_logging.get_verbosity()
        tf_logging.set_verbosity(tf_logging.ERROR)

        ret = func(*args, **kwargs)

        # Restore logging settings
        tf_logging.set_verbosity(verbosity)

        return ret
    return wrapper


def default_saved_model_tags():
    return [tag_constants.SERVING]


def removable_graph_keys():
    return ops.GraphKeys._VARIABLE_COLLECTIONS + \
        [
            ops.GraphKeys._SUMMARY_COLLECTION,
            ops.GraphKeys.SUMMARIES,
            ops.GraphKeys.SUMMARY_OP,
            ops.GraphKeys.GLOBAL_STEP,
        ]


def try_default_feed_dict(name):
    exact_feed_dict = {
        "dropout": 1.0
    }
    vague_feed_dict = {
        "learn": False,
        "train": False,
        "keep_prob": 1.0,
    }
    if name in exact_feed_dict:
        return exact_feed_dict[name]

    for k in vague_feed_dict:
        if k in name:
            return vague_feed_dict[k]
    return None


def get_node_name(full_name):
    """Get node actual name from full name in graph_def connection"""
    is_ctrl = False
    port = -1
    node_name = full_name
    if full_name.startswith('^'):
        node_name = full_name[1:]
        is_ctrl = True

    node_name_split = node_name.split(':')
    node_name = node_name_split[0]
    if len(node_name_split) > 1:
        port = int(node_name_split[1])
    return node_name, is_ctrl, port


def get_value_from_const(node):
    return tf.make_ndarray(node.attr["value"].tensor)


def make_const_node(name, dtype, data):
    if not isinstance(dtype, tf.DType):
        dtype = tf.DType(dtype)
    if not isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=dtype.as_numpy_dtype)
    node = tf.NodeDef()
    node.op = "Const"
    node.name = name
    node.attr["dtype"].type = dtype.as_datatype_enum
    node.attr["value"].CopyFrom(tf.AttrValue(
        tensor=tf.contrib.util.make_tensor_proto(
            data, dtype=dtype.as_datatype_enum, shape=data.shape)))
    return node


def make_gather_node(node_name, weight, query, axis, taxis, tid, tparam, batch_dim):
    # create gather node
    new_node = tf.NodeDef()
    new_node.name = node_name
    new_node.op = 'GatherV2'
    new_node.input.append(weight)
    new_node.input.append(query)
    new_node.input.append(axis)
    new_node.attr['Taxis'].type = taxis
    new_node.attr['Tindices'].type = tid
    new_node.attr['Tparams'].type = tparam
    new_node.attr['batch_dims'].i = batch_dim
    return new_node


def make_slice_node(op_name, input_name, begin_name, end_name, data_type=3, id_type=3):
    # (todo:xlx) this function should be moved to utils
    new_node = tf.NodeDef()
    new_node.op = 'Slice'
    new_node.attr['T'].type = data_type
    new_node.attr['Index'].type = id_type
    new_node.name = op_name
    new_node.input.append(input_name)
    new_node.input.append(begin_name)
    new_node.input.append(end_name)
    return new_node


def assert_const_value(node, value):
    """assert const node contains certain value"""
    return node is not None and value == get_value_from_const(node)


def _get_op_tensor_with_key(meta_graph_def_to_load, op_key):
    """Gets the main op tensor, if one exists.

    Args:
      meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.

    Returns:
      The main op tensor, if it exists and `None` otherwise.

    Raises:
      RuntimeError: If the collection def corresponding to the main op key has
          other than exactly one tensor.
    """
    main_op_keys = [constants.MAIN_OP_KEY, constants.LEGACY_INIT_OP_KEY]
    collection_def = meta_graph_def_to_load.collection_def
    if op_key in collection_def:
        op_tensors = collection_def[op_key].node_list.value
        if op_key in main_op_keys and len(op_tensors) != 1:
            raise RuntimeError("Expected exactly one SavedModel main op.")
        return op_tensors
    return None


def get_main_op_tensor(meta_graph_def):
    main_op_tensor = _get_op_tensor_with_key(meta_graph_def, constants.MAIN_OP_KEY) or \
        _get_op_tensor_with_key(meta_graph_def, constants.LEGACY_INIT_OP_KEY)
    return None if not main_op_tensor else main_op_tensor[0]


def _get_asset_tensors(meta_graph_def, saved_model_dir=None):
    """Gets the asset tensors, if defined in the meta graph def to load.
    Args:
        meta_graph_def: The meta graph def from the SavedModel to be loaded.
    Returns:
        A dictionary of asset tensors, keyed by the name of the asset tensor. The
        value in the map corresponds to the absolute path of the asset file.
    """
    # Collection-def that may contain the assets key.
    collection_def = meta_graph_def.collection_def

    asset_tensor_dict = {}
    asset_protos = []

    if meta_graph_def.asset_file_def:
        asset_protos = meta_graph_def.asset_file_def
    elif constants.ASSETS_KEY in collection_def:
        assets_any_proto = collection_def[constants.ASSETS_KEY].any_list.value
        for asset_any_proto in assets_any_proto:
            asset_proto = meta_graph_pb2.AssetFileDef()
            asset_any_proto.Unpack(asset_proto)
            asset_protos.append(asset_proto)

    if saved_model_dir:
        assets_dir = os.path.join(
            compat.as_bytes(saved_model_dir), compat.as_bytes(constants.ASSETS_DIRECTORY))

    # Process each asset and add it to the asset tensor dictionary.
    for asset_proto in asset_protos:
        tensor_name = asset_proto.tensor_info.name
        asset_tensor_dict[tensor_name] = os.path.join(
            compat.as_bytes(assets_dir), compat.as_bytes(asset_proto.filename)) \
            if saved_model_dir else None

    return asset_tensor_dict


def remove_nodes_from_graph(graph_def, removable_nodes):
    new_gd = tf.GraphDef()
    new_nodes = []
    for n in graph_def.node:
        if n.name in removable_nodes:
            continue
        new_n = tf.NodeDef()
        new_n.CopyFrom(n)
        del new_n.input[:]
        for iedge in n.input:
            inode_name, _, _ = get_node_name(iedge)
            if inode_name not in removable_nodes:
                new_n.input.append(iedge)
        new_nodes.append(new_n)
    new_gd.node.extend(new_nodes)
    return new_gd


def get_meta_graph_io(meta_graph_def):
    input_nodes = {}
    output_nodes = {}
    for v in meta_graph_def.signature_def.values():
        for k, tensor in v.inputs.items():
            input_nodes[k] = tensor.name
        for k, tensor in v.outputs.items():
            output_nodes[k] = tensor.name
    return input_nodes, output_nodes


def create_new_meta_graph_def(meta_graph_def,
                              new_graph_def, inputs):
    r"""Update meta_graph_def with new graph_def
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/meta_graph.proto

    Params
    ------
        meta_graph_def: tf.MetaGraphDef
            Original meta graph that contains an old GraphDef and meta information
        new_graph_def: tf.GraphDef
            New GraphDef that comes from the old GraphDef, possibly optimized or frozen
    Returns
    -------
        mgd: tf.MetaGraphDef
            New MetaGraphDef that contains the new GraphDef
    """
    
    #print(new_graph_def)

   
    new_nodes = [node.name for node in new_graph_def.node]
    old_nodes = [node.name for node in meta_graph_def.graph_def.node]
    exclude_nodes = []
    for i in old_nodes:
      if i not in new_nodes:
        exclude_nodes.append(i)  

    mgd = meta_graph.create_meta_graph_def(
        meta_info_def=meta_graph_def.meta_info_def,
        graph_def=new_graph_def,
        graph=tf.Graph(),
        saver_def=meta_graph_def.saver_def,
        exclude_nodes=exclude_nodes)

    new_nodes = {n.name: n.op for n in new_graph_def.node}
    asset_nodes = _get_asset_tensors(meta_graph_def)
    keep_nodes = list(new_nodes.keys()) #+ list(asset_nodes.keys())
     
    

    for k, v in meta_graph_def.signature_def.items():
        removable_sig = set()
        for tname, tensor in v.inputs.items():
            nname = get_node_name(tensor.name)[0]
            if nname not in new_nodes:
                logger.warning("Input %s no longer exists", tname)
                removable_sig.add(tname)
                sample = tensor
            elif new_nodes[nname] == "Const":
                removable_sig.add(tname)
        mgd.signature_def[k].CopyFrom(v)
        for tname in removable_sig:
            mgd.signature_def[k].inputs.pop(tname)
        for key, node in inputs.items():
            tensor_info = meta_graph_pb2.TensorInfo(dtype=dtypes.as_dtype(node.attr["dtype"].type).as_datatype_enum, 
        tensor_shape=node.attr["shape"].shape)
            tensor_info.name = node.name
            mgd.signature_def[k].inputs[key].CopyFrom(tensor_info)
        

        for tname, tensor in v.outputs.items():
            nname = get_node_name(tensor.name)[0]
            if nname not in new_nodes:
                logger.warning("Output %s no longer exists", tname)

    updatable_bytes_list_types = [ops.GraphKeys.COND_CONTEXT]
    for k, v in meta_graph_def.collection_def.items():
        if k == constants.ASSETS_KEY:
            continue

        if k in removable_graph_keys() and k != ops.GraphKeys.GLOBAL_VARIABLES:
            continue
            #pass

        if k == ops.GraphKeys.GLOBAL_VARIABLES:
          #print(v)
          pass
        mgd.collection_def[k].CopyFrom(v)
        kind = v.WhichOneof("kind")

        # (TODO: support more collections)

        # kind other than node_list or bytes_list, keep as it is
        if kind not in {"node_list", "bytes_list"}:
            continue

        # process node_list
        if kind == "node_list":
            del mgd.collection_def[k].node_list.value[:]
            for op in v.node_list.value:
                op_name = get_node_name(op)[0]
                if op_name in keep_nodes:
                    mgd.collection_def[k].node_list.value.append(op)
      
        
        if k in ops.GraphKeys._VARIABLE_COLLECTIONS:
            proto_type = ops.get_collection_proto_type(k)
            del mgd.collection_def[k].bytes_list.value[:]
            for value in v.bytes_list.value:
                #print("value " + str(value))
                proto = proto_type()
                proto_new = proto_type()
                proto.ParseFromString(value)
                proto_new.ParseFromString(value)
                pivot_node = get_node_name(proto.variable_name)[0]
                if pivot_node not in exclude_nodes:
                    proto_new.variable_name = proto.variable_name
                    if get_node_name(proto.initial_value_name)[0] not in exclude_nodes:
                      proto_new.initial_value_name = proto.initial_value_name
                    else:
                      proto_new.initial_value_name = proto.variable_name
                    if get_node_name(proto.snapshot_name)[0] not in exclude_nodes:
                      proto_new.snapshot_name = proto.snapshot_name
                    else:
                      proto_new.snapshot_name = proto.variable_name
                    if get_node_name(proto.initializer_name)[0] not in exclude_nodes:
                      proto_new.initializer_name = proto.initializer_name
                    else:
                      proto_new.initializer_name = proto.variable_name
                    value_new = proto_new.SerializeToString()
                    mgd.collection_def[k].bytes_list.value.append(value_new)
        
        # process bytes_list
        # unrecognizable proto, keep as it is
        from_proto = ops.get_from_proto_function(k)
        if not from_proto or k not in updatable_bytes_list_types:
            continue
         

        # process cond_text bytes_list
        elif k == ops.GraphKeys.COND_CONTEXT:
            proto_type = ops.get_collection_proto_type(k)
            del mgd.collection_def[k].bytes_list.value[:]
            for value in v.bytes_list.value:
                proto = proto_type()
                proto.ParseFromString(value)
                pivot_node = get_node_name(proto.pivot_name)[0]
                if pivot_node in keep_nodes:
                    mgd.collection_def[k].bytes_list.value.append(value)

    for asset in meta_graph_def.asset_file_def:
        mgd.asset_file_def.append(asset)

    # object_graph_def is added after tf 1.14
    if hasattr(mgd, 'object_graph_def'):
        mgd.object_graph_def.CopyFrom(meta_graph_def.object_graph_def)

    #print(mgd)

    return mgd


def match_tf_version(v1, v2):
    """Match tensorflow version string

    Params
    ------
    v1: str
        tf version string
    v2: str
        tf version string

    Returns
    -------
    bool
        Only return False if platform mismatch,
        one with 'PAI' and one without
    """
    if ('PAI' in v1) ^ ('PAI' in v2):
        logger.warning("TF Platform Mismatch!(%s %s)", v1, v2)
        return False
    if v1 != v2:
        logger.warning("TF Version Mismatch!(%s %s)", v1, v2)
        return True
    return True


def get_tf_version():
    return tf.__version__


def get_tf_registered_ops():
    return set(tf.python.ops.op_def_registry.get_registered_ops().keys())


def set_attr_f(node, key, value):
    """Set float attribute"""
    try:
        node.attr[key].CopyFrom(tf.AttrValue(f=value))
    except KeyError:
        pass


def set_attr_b(node, key, value):
    """Set string attribute"""
    try:
        node.attr[key].CopyFrom(tf.AttrValue(b=value))
    except KeyError:
        pass


def set_attr_i(node, key, value):
    """Set int attribute"""
    try:
        node.attr[key].CopyFrom(tf.AttrValue(i=value))
    except KeyError:
        pass


def check_node_rank(node, rank):
    # get shapes
    if "_output_shapes" not in node.attr:
        return False
    if not node.attr["_output_shapes"].list.shape:
        return False
    if len(node.attr["_output_shapes"].list.shape[0].dim) != rank:
        return False
    return True
