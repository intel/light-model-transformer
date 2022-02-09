# Copyright 2020 TF-STK Authors. All Rights Reserved.
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
"""tvm utilities"""

import tensorflow as tf

# commit bd17baa215d37de045a32aaa08112f509fdc2633
tvm_ops = [
    'Placeholder'                       ,
    'PlaceholderWithDefault'            ,
    'Const'                             ,

    'Abs'                               ,
    'Add'                               ,
    'AddN'                              ,
    'All'                               ,
    'Any'                               ,
    'ArgMax'                            ,
    'ArgMin'                            ,
    'Assert'                            ,
    'AvgPool'                           ,
    'AvgPool3D'                         ,
    'BatchMatMul'                       ,
    'BatchMatMulV2'                     ,
    'BatchNormWithGlobalNormalization'  ,
    'BatchToSpaceND'                    ,
    'BiasAdd'                           ,
    'BroadcastTo'                       ,
    'Cast'                              ,
    'Ceil'                              ,
    'CheckNumerics'                     ,
    'ClipByValue'                       ,
    'Concat'                            ,
    'ConcatV2'                          ,
    'Conv2D'                            ,
    'Conv3D'                            ,
    'Conv2DBackpropInput'               ,
    'CropAndResize'                     ,
    'DecodeJpeg'                        ,
    'DepthwiseConv2dNative'             ,
    'DepthToSpace'                      ,
    'Equal'                             ,
    'Elu'                               ,
    'Erf'                               ,
    'Exp'                               ,
    'ExpandDims'                        ,
    'Fill'                              ,
    'Floor'                             ,
    'FloorDiv'                          ,
    'FloorMod'                          ,
    'FusedBatchNorm'                    ,
    'FusedBatchNormV2'                  ,
    'Gather'                            ,
    'GatherNd'                          ,
    'GatherV2'                          ,
    'Greater'                           ,
    'GreaterEqual'                      ,
    'Identity'                          ,
    'LeakyRelu'                         ,
    'LeftShift'                         ,
    'Less'                              ,
    'LessEqual'                         ,
    'Log'                               ,
    'Log1p'                             ,
    'Cos'                               ,
    'Sin'                               ,
    'LogicalAnd'                        ,
    'LogicalOr'                         ,
    'LogicalNot'                        ,
    'LogSoftmax'                        ,
    'LRN'                               ,
    'MatMul'                            ,
    'Max'                               ,
    'MaxPool'                           ,
    'MaxPool3D'                         ,
    'Maximum'                           ,
    'Mean'                              ,
    'Min'                               ,
    'Minimum'                           ,
    'MirrorPad'                         ,
    'Mod'                               ,
    'Mul'                               ,
    'Neg'                               ,
    'NoOp'                              ,
    'NotEqual'                          ,
    'OneHot'                            ,
    'Pack'                              ,
    'TensorArrayV3'                     ,
    'TensorArrayScatterV3'              ,
    'TensorArrayGatherV3'               ,
    'TensorArraySizeV3'                 ,
    'TensorArrayWriteV3'                ,
    'TensorArrayReadV3'                 ,
    'TensorArraySplitV3'                ,
    'TensorArrayConcatV3'               ,
    'Pad'                               ,
    'PadV2'                             ,
    'Pow'                               ,
    'Prod'                              ,
    'Range'                             ,
    'Rank'                              ,
    'RealDiv'                           ,
    'Relu'                              ,
    'Relu6'                             ,
    'Reshape'                           ,
    'ResizeBilinear'                    ,
    'ResizeBicubic'                     ,
    'ResizeNearestNeighbor'             ,
    'ReverseV2'                         ,
    'RightShift'                        ,
    'Round'                             ,
    'Rsqrt'                             ,
    'Select'                            ,
    'Selu'                              ,
    'Shape'                             ,
    'Sigmoid'                           ,
    'Sign'                              ,
    'Size'                              ,
    'Slice'                             ,
    'Softmax'                           ,
    'Softplus'                          ,
    'SpaceToBatchND'                    ,
    'SpaceToDepth'                      ,
    'Split'                             ,
    'SplitV'                            ,
    'Sqrt'                              ,
    'Square'                            ,
    'SquaredDifference'                 ,
    'Squeeze'                           ,
    'StopGradient'                      ,
    'StridedSlice'                      ,
    'Sub'                               ,
    'Sum'                               ,
    'Tanh'                              ,
    'Tile'                              ,
    'TopKV2'                            ,
    'Transpose'                         ,
    'TruncateMod'                       ,
    'Unpack'                            ,
    'Where'                             ,
    'ZerosLike'                         ,

    'Merge'                             ,
    'Switch'                            ,
    'NextIteration'                     ,
    'Exit'                              ,
    'Enter'                             ,
    'LoopCond'                          ,
    'LSTMBlockCell'                     ,
]

def check_graph_op_coverage(graph):
    """Check if tf graph ops are covered by tvm

    Params
    ------
    graph: tf.Graph or tf.GraphDef
        The tensorflow graph to be checked

    Returns
    -------
    unsupported: set(str)
        The operations in graph not covered by tvm
    """
    if isinstance(graph, tf.Graph):
        graph = graph.as_graph_def()
    assert(isinstance(graph, tf.GraphDef))
    unsupported = {}
    cover_count = len(graph.node)
    for n in graph.node:
        if n.op not in tvm_ops:
            cover_count -= 1
            if n.op in unsupported:
                unsupported[n.op].append(n.name)
            else:
                unsupported[n.op] = [n.name]
    cover_rate = 100 * (cover_count / len(graph.node))
    print("Cover Rate %f%%" % (cover_rate))
    return unsupported

