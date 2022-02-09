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
import os
import shutil
import random

from google.protobuf import text_format
import tensorflow as tf
print(tf.__version__)
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import meta_graph
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.training import saver as tf_saver
try:
    from tensorflow.contrib.saved_model.python.saved_model.reader import read_saved_model
except ImportError:
    try:
        from tensorflow.python.tools.saved_model_utils import read_saved_model
    except ImportError:
        raise RuntimeError("tfstk requires tensorflow >= 1.12, please check")

from . import tf_utils
from . import stk_logging
logger = stk_logging.get_logger()


def dump_frozen_pb(dump_dir, file_name, graph_def, as_text=False):
    tf.io.write_graph(graph_def, dump_dir, file_name, as_text)


def dump_saved_model(dump_dir, meta_graph_def, tags):

    # tf.load_op_library("/disk1/workspace/eas_top7/tf-stk/python/tests/custom_ops/bert_op.so")
    tmp_dir = str(random.getrandbits(128))
    if os.path.isdir(dump_dir):
        shutil.copytree(dump_dir, tmp_dir)
        shutil.rmtree(dump_dir, ignore_errors=True)

    builder = tf.saved_model.builder.SavedModelBuilder(dump_dir)
    #print(meta_graph_def)

    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph.import_scoped_meta_graph(meta_graph_def)
        """
        ss, _ = tf_saver._import_meta_graph_with_return_elements( 
            meta_graph_def)
        ss.restore(sess, "./coconut_2021/variables/variables") 
        init_op = loader_impl.get_init_op(meta_graph_def)
        if init_op is not None:
          print("init")
          sess.run(fetches=[init_op])"""
        #builder.add_meta_graph_and_variables(
        builder._has_saved_variables = True
        builder.add_meta_graph(
            #sess,
            tags,
            meta_graph_def.signature_def,
            meta_graph_def.asset_file_def)
    builder.save()

    if os.path.isdir(tmp_dir):
        assets_src = os.path.join(tmp_dir, "assets")
        assets_dst = os.path.join(dump_dir, "assets")
        assets_extra_src = os.path.join(tmp_dir, "assets.extra")
        assets_extra_dst = os.path.join(dump_dir, "assets.extra")
        if os.path.isdir(assets_src):
            shutil.rmtree(assets_dst, ignore_errors=True)
            shutil.copytree(assets_src, assets_dst)
        if os.path.isdir(assets_extra_src):
            shutil.rmtree(assets_extra_dst, ignore_errors=True)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def dump_new_saved_model(new_dir, old_dir, new_mgd, tags):
    """Dump new meta graph def with original assets to new dir

    Params
    ------
    new_dir: str
        New directory to dump meta graph def to
    old_dir: str
        Old directory that possibly contains asset files
    new_mgd: tf.MetaGraphDef
        Meta graph def that has been frozen/optimized etc.
    """
    dump_saved_model(new_dir, new_mgd, tags)

    if os.path.abspath(old_dir) != os.path.abspath(new_dir):
        assets_src = os.path.join(old_dir, "assets")
        assets_dst = os.path.join(new_dir, "assets")
        assets_extra_src = os.path.join(old_dir, "assets.extra")
        assets_extra_dst = os.path.join(new_dir, "assets.extra")
        if os.path.isdir(assets_src):
            shutil.rmtree(assets_dst, ignore_errors=True)
            shutil.copytree(assets_src, assets_dst)
        if os.path.isdir(assets_extra_src):
            shutil.rmtree(assets_extra_dst, ignore_errors=True)
            shutil.copytree(assets_extra_src, assets_extra_dst)
    else:
        logger.warning("New/old saved model folders are the same: %s",
                       os.path.abspath(old_dir))


def load_meta_graph_def(saved_model_dir, tags, ignore_version=True):

    if not loader.maybe_saved_model_directory(saved_model_dir):
        raise IOError("Saved model not found in %s" % saved_model_dir)

    saved_model_pb = read_saved_model(saved_model_dir)
    if (len(saved_model_pb.meta_graphs) > 1):
        logger.warning(
            "Found more than one meta graphs in saved model dir, choosing 1st one")

    tag_set = set(tags)
    for mgd in saved_model_pb.meta_graphs:
        if set(mgd.meta_info_def.tags) == tag_set:
            if not ignore_version:
                tf_utils.match_tf_version(
                    mgd.meta_info_def.tensorflow_version, tf.__version__)
            return mgd

    raise RuntimeError("MetaGraphDef associated with tag-set " + str(tag_set) +
                       " could not be found in " + saved_model_dir)


def load_graph_def_from_saved_model(saved_model_dir,
                                    tags=None,
                                    output_nodes=None,
                                    ignore_version=True):
    output_nodes = [] if output_nodes is None else output_nodes
    mgd = load_meta_graph_def(saved_model_dir, tags, ignore_version)

    for sig in mgd.signature_def.values():
        output_nodes += [v.name.split(':')[0]
                         for k, v in sig.outputs.items()]

    keep_collections = {constants.MAIN_OP_KEY,
                        constants.LEGACY_INIT_OP_KEY}
    keep_nodes = set()
    for name, collection in mgd.collection_def.items():
        if name in keep_collections:
            keep_nodes |= {tf_utils.get_node_name(op)[0]
                           for op in collection.node_list.value}
    keep_nodes |= tf_utils._get_asset_tensors(mgd).keys()
    output_nodes += keep_nodes

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        loader.load(sess, {tags}, saved_model_dir)
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        graph_def = graph_util.convert_variables_to_constants(
            sess, graph_def, output_nodes)

    return graph_def


def load_graph_def(graph_pb_file, is_binary=True):
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(graph_pb_file, "rb") as f:
        if is_binary:
            graph_def.ParseFromString(f.read())
        else:
            graph_def = text_format.Parse(f.read(), graph_def)

    return graph_def


# def load_checkpoint(ckpt_dir, prefix, ignore_version=True):
#    prefix = os.path.join(ckpt_dir, prefix)
#    if not checkpoint_management.checkpoint_exists(prefix):
#        raise IOError("Checkpoint not found in %s" % prefix)
#
#    meta_file = prefix + ".meta"
#    with tf.GFile(meta_file, "rb") as f:
#        mgd = tf.MetaGraphDef()
#        mgd.ParseFromString(f.read())
#
#    return mgd
