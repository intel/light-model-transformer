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
"""Utility class/functions for tf saved models"""
import os
import time
from enum import Enum
import numpy as np

import tensorflow as tf
print(tf.__version__)
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
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


class SavedModelMode(Enum):
    FULL = 1
    COMPATIBLE = 2
    INCOMPATIBLE = 3

# pylint: disable=too-many-instance-attributes
class SavedModelInfo:
    """Saved model information class"""

    def __init__(self, saved_model_dir, tags):
        self._saved_model_dir = saved_model_dir
        self._tags = tags
        self._input_mgd_version = None

        self._mode = SavedModelMode.FULL
        self._mgd = None
        self._compatible_mgd = None
        self._frozen_mgd = None

        self._input_sig = {}
        self._output_sig = {}
        self._all_nodes = []
        self._infer_nodes = []
        self._all_ops = set()
        self._infer_ops = set()
        self._invalid_ops = set()
        self._invalid_full_ops = set()

        self._load_saved_model()

        # owns a session for saved model freezing and dumping
        self._sess = tf.Session(graph=tf.Graph())

    @property
    def origin_mgd(self):
        return self._mgd

    @property
    def compatible_mgd(self):
        return self._compatible_mgd

    @property
    def frozen_mgd(self):
        return self._frozen_mgd

    @property
    def mode(self):
        return self._mode

    @property
    def invalid_ops(self):
        return self._invalid_ops

    @property
    def invalid_infer_ops(self):
        return self._invalid_infer_ops

    def _load_saved_model(self):
        """Load saved model from dir.

        Note
        ----
        Session NOT INVOLVED

        """
        if not loader.maybe_saved_model_directory(self._saved_model_dir):
            raise IOError("Saved model not found in {}".format(
                self._saved_model_dir))

        saved_model_pb = read_saved_model(self._saved_model_dir)

        tag_set = set(self._tags)
        for mgd in saved_model_pb.meta_graphs:
            if set(mgd.meta_info_def.tags) == tag_set:
                self._mgd = mgd
                break

        if not self._mgd:
            raise RuntimeError("MetaGraphDef associated with tag-set " + str(tag_set) +
                               " could not be found in " + self._saved_model_dir)

        self._input_mgd_version = self._mgd.meta_info_def.tensorflow_version
        tf_utils.match_tf_version(
            self._input_mgd_version, tf_utils.get_tf_version())

        for v in self._mgd.signature_def.values():
            for k, tensor in v.inputs.items():
                self._input_sig[k] = tensor.name
            for k, tensor in v.outputs.items():
                self._output_sig[k] = tensor.name

        input_nodes = [tf_utils.get_node_name(
            v)[0] for v in self._input_sig.values()]
        output_nodes = [tf_utils.get_node_name(
            v)[0] for v in self._output_sig.values()]

        # reserve init ops
        print("*********************")
        main_op_node = tf_utils.get_main_op_tensor(self._mgd)
        if main_op_node:
            output_nodes.append(tf_utils.get_node_name(main_op_node)[0])

        graph_def = self._mgd.graph_def
        for n in graph_def.node:
            self._all_ops.add(n.op)
            self._all_nodes.append(n.name)
            # For compatibility of PAI-TF models
            if n.op == "SaveV2":
                n.attr.pop('has_ev')
            n.attr.pop('_output_shapes')

        output_pre_gd = graph_util.extract_sub_graph(graph_def, output_nodes)
        input_pre_gd = graph_util.extract_sub_graph(graph_def, input_nodes)

        # verify against current tf op set
        for n in output_pre_gd.node:
            if n in input_pre_gd.node and n.name not in input_nodes:
                continue
            self._infer_nodes.append(n.name)
            self._infer_ops.add(n.op)

        if tf.__version__[0] == '1':
            tf_op_set = tf_utils.get_tf_registered_ops()
            self._invalid_ops = self._all_ops - tf_op_set
            self._invalid_infer_ops = self._infer_ops - tf_op_set
        else:
            self._invalid_ops = set()
            self._invalid_infer_ops = set()

        # determine compatibility mode of saved model against current tf
        if self._invalid_infer_ops:
            self._mode = SavedModelMode.INCOMPATIBLE
        elif self._invalid_ops:
            new_gd = tf.GraphDef()
            new_nodes = []
            invalid_node_names = []
            for n in graph_def.node:
                if n.op in self._invalid_ops:
                    invalid_node_names.append(n.name)

            for n in graph_def.node:
                if n.op in self._invalid_ops:
                    continue
                new_node = tf.NodeDef()
                new_node.CopyFrom(n)

                del new_node.input[:]
                for ni in n.input:
                    ni_name = tf_utils.get_node_name(ni)[0]
                    if ni_name not in invalid_node_names:
                        new_node.input.append(ni)
                new_nodes.append(new_node)
            new_gd.node.extend(new_nodes)
            self._compatible_mgd = tf_utils.create_new_meta_graph_def(
                self._mgd, new_gd)
            self._mode = SavedModelMode.COMPATIBLE
        else:
            self._compatible_mgd = self._mgd
            self._mode = SavedModelMode.FULL

    def freeze(self):
        """Freeze saved model

        Note
        ----
        Session INVOLVED
        """
        if self._mode == SavedModelMode.INCOMPATIBLE:
            raise RuntimeError(
                "Freezing saved model({}) failed. "
                "Ops unsupported by current tf({}): {}".format(self._input_mgd_version,
                                                               tf_utils.get_tf_version(),
                                                               str(self._invalid_ops)))

        with self._sess.graph.as_default():
            saver = tf_saver.import_meta_graph(self._compatible_mgd)
            if not saver:
                logger.warning(
                    "Saved model is already frozen as no variable is found in the model")
                self._frozen_mgd = self._compatible_mgd
                return

            variable_path = os.path.join(
                compat.as_bytes(self._saved_model_dir),
                compat.as_bytes(constants.VARIABLES_DIRECTORY),
                compat.as_bytes(constants.VARIABLES_FILENAME)
            )
            saver.restore(self._sess, variable_path)

            input_nodes = [tf_utils.get_node_name(
                v)[0] for v in self._input_sig.values()]
            output_nodes = [tf_utils.get_node_name(
                v)[0] for v in self._output_sig.values()]

            # reserve unused input placeholders
            output_nodes += input_nodes

            # reserve init ops
            main_op_node = tf_utils.get_main_op_tensor(self._mgd)
            if main_op_node:
                output_nodes.append(tf_utils.get_node_name(main_op_node)[0])

            # Get asset tensors, if any.
            asset_tensors_dictionary = tf_utils._get_asset_tensors(
                self._mgd, self._saved_model_dir)

            if asset_tensors_dictionary:
                output_nodes += [tf_utils.get_node_name(k)[0]
                                 for k in asset_tensors_dictionary]

            if main_op_node is not None:
                self._sess.run(fetches=[main_op_node],
                               feed_dict=asset_tensors_dictionary)

            frozen_gd = graph_util.convert_variables_to_constants(
                self._sess, self._compatible_mgd.graph_def, output_nodes)

        self._frozen_mgd = tf_utils.create_new_meta_graph_def(
            self._mgd, frozen_gd)


def _auto_fill_numeric_tensor(shape, dtype, seed):
    np.random.seed(seed)
    if dtype.is_floating:
        return np.random.rand(*shape).astype(dtype.as_numpy_dtype)
    if dtype.is_integer:
        return np.random.randint(0, 80, shape, dtype.as_numpy_dtype)
    if dtype.is_bool:
        return np.random.randint(2, shape, np.bool)
    return None


# pylint: disable=too-many-locals
def benchmark_saved_model(saved_model_dir, tags, min_repeat_ms=20000, bs=1, seed=0, inter=0, intra=0, feed_dict=None):

    config = tf.ConfigProto(inter_op_parallelism_threads=inter,
                            intra_op_parallelism_threads=intra,
                            use_per_session_threads=True)

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        mgd = loader.load(sess, tags, saved_model_dir)
        fetch_dict = {}
        for _, ioconfig in mgd.signature_def.items():
            if feed_dict is None:
                feed_dict = {}
                input_map = dict(ioconfig.inputs)
                for k, v in input_map.items():
                    shape = tuple(bs if dim.size <
                                  0 else dim.size for dim in v.tensor_shape.dim)
                    tf_dtype = tf.DType(v.dtype)
                    if tf_dtype.is_numpy_compatible:
                        feed_dict[v.name] = _auto_fill_numeric_tensor(
                            shape, tf_dtype, seed)
                    else:
                        # TODO
                        raise RuntimeError("{} does not support auto fill")

            output_map = dict(ioconfig.outputs)
            for k, v in output_map.items():
                fetch_dict[k] = v.name

        st = time.time()
        # warmup
        for _ in range(3):
            _ = sess.run(fetch_dict, feed_dict=feed_dict)
        et = (time.time() - st) * 1000
        at = et / 3

        time_left = min_repeat_ms
        total_runs = 0
        st_gold = time.time()
        while time_left > 0:
            run_number = int(time_left / at * 1.236)
            st = time.time()
            for _ in range(run_number):
                _ = sess.run(fetch_dict, feed_dict=feed_dict)
            et = (time.time() - st) * 1000
            time_left -= et
            at = et / run_number
            total_runs += run_number
        et_gold = time.time() - st_gold
        at_gold = et_gold * 1000 / total_runs
        print("Model Location: {}".format(saved_model_dir))
        print("Model Time    : {0:4f}ms, total {1} runs in {2:4f}s".format(
              at_gold, total_runs, et_gold))
        print("TF Version    : {}".format(tf.__version__))
