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
import tensorflow as tf
import numpy as np
import sys
if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle



def _fetch_const(sess, name):
    return sess.run(name+":0")


def has_const_input(c, name_node_dict):
    flag = False
    wait_checked = [e for e in c if c.get(e) == False ]
    for e in wait_checked:
        e = name_node_dict.get(e)
        if e.input and all(map(lambda x: c.get(x), e.input)):
            c[e.name] = True
            flag = True
    return flag, c


def get_consts_names(pb_path):
    result = []
    output_graph_path = pb_path

    with tf.Graph().as_default() as g:
        output_graph_def = tf.GraphDef()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        const_dict = {}
        name_op_dict = {}
        name_node_dict = {}
        for node in g.as_graph_def().node:
            const_dict[node.name] = False
            name_op_dict[node.name] = node.op
            name_node_dict[node.name] = node
        # init direct const node
        for node_name in name_node_dict:
            if name_op_dict.get(node_name) == "Const" and not name_node_dict.get(node_name).input:
                const_dict[node_name] = True
        while 1:
            flag, const_dict = has_const_input(const_dict, name_node_dict)
            if not flag:
                break
    result = [e for e in const_dict if const_dict.get(e)]
    return result


def get_consts(pb_path):
    result = {}
    const_names = get_consts_names(pb_path)
    output_graph_path = pb_path
    with tf.Graph().as_default() as g:
        output_graph_def = tf.GraphDef()
        with open(output_graph_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                for name in const_names:
                    try:
                        result[name] = _fetch_const(sess, name)
                    except Exception as e:
                        print("\tIgnore possible const node: %s" % name)

    return result
