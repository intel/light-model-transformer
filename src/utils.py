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
import numpy as np
import logging

def get_logger():
    LOG_FORMAT = 'LINE:%(lineno)s - %(levelname)s: %(message)s'
    logging.basicConfig(format=LOG_FORMAT)
    logger = logging.getLogger("convert_logger")
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()
g_weights = []

# get_str(b'NHWC') = 'NHWC'
def decode_string(str):
    str = str.decode('UTF-8')
    return str


def do_multiply(conv_node, weights, multiplier):
    if conv_node.op == 'Conv2D':
        if weights.shape[3] != multiplier.shape[0]:
            print("Error: mismatched shape, weights.shape=")
            print(weights.shape)
            print("multiplier.shape=")
            print(multiplier.shape)
            exit(-1)
        for k_h in range(0, weights.shape[0]):
            for k_w in range(0, weights.shape[1]):
                for i_c in range(0, weights.shape[2]):
                    for o_c in range(0, weights.shape[3]):
                        weights[k_h][k_w][i_c][o_c] *= multiplier[o_c]

    elif conv_node.op == 'DepthwiseConv2dNative':
        if (weights.shape[2] * weights.shape[3]) != multiplier.shape[0]:
            print("Error: mismatched shape, weights.shape=")
            print(weights.shape)
            print("multiplier.shape=")
            print(multiplier.shape)
            exit(-1)
        for k_h in range(0, weights.shape[0]):
            for k_w in range(0, weights.shape[1]):
                idx = 0
                for i_c in range(0, weights.shape[2]):
                    for o_c in range(0, weights.shape[3]):
                        weights[k_h][k_w][i_c][o_c] *= multiplier[idx]
                        idx += 1
    return weights


# weights is 2 dimaension tensor
# It is currently called to merge 'Matmul' and 'Mul' (our implementation is not so rigorous)
def do_multiply2(weights, multiplier):
    if weights.shape[0] == multiplier.shape[0]:
        for o in range(0, weights.shape[0]):
            for i in range(0, weights.shape[1]):
                weights[o][i] *= multiplier[o]

    elif weights.shape[1] == multiplier.shape[0]:
        for i in range(0, weights.shape[0]):
            for o in range(0, weights.shape[1]):
                weights[i][o] *= multiplier[o]
    else:
        print("Error: mismatched shape, weights.shape=")
        print(weights.shape)
        print("multiplier.shape=")
        print(multiplier.shape)
        exit(-1)


def write_to_file(f, str):
    if sys.version_info[0] == 3:
        f.write(str.encode())
    else:
        f.write(str)


def dump_fake_input(height, width, channel, topo_file):
    write_to_file(topo_file, "# op origin_name out_channel height width\n")
    write_to_file(topo_file, "Placeholder Placeholder %s %s %s\n" % (channel, height, width))


def dump_placeholder(name, topo_file, height, width, channel):
    write_to_file(topo_file, "# op origin_name out_channel height width\n")
    write_to_file(topo_file, "Placeholder %s %d %d %d\n" % (name ,channel, height, width))


def dump_merged_op(node, topo_file):
    write_to_file(topo_file, "# Merged op: %s\n" % node.op)


def dump_ignored_op(node, topo_file):
    write_to_file(topo_file, "# Ignored op: %s, can be ignored?\n" % node.op)


def dump_unsupported_op(node, topo_file):
    write_to_file(topo_file, "*** Unsupported op: %s ***\n" % node.op)


def dump_simple_op(node, name, input_name, topo_file):
    write_to_file(topo_file, "# op origin_name input_name\n")
    ipt_name = " ".join(input_name) if isinstance(
        input_name, list) else input_name
    write_to_file(topo_file, "%s %s %s\n" % (node.op, name, ipt_name))


def dump_relu(node, name, input_name, topo_file):
    write_to_file(topo_file, "# op origin_name input_name alpha beta\n")
    ipt_name = " ".join(input_name) if isinstance(
        input_name, list) else input_name

    try:
        str_alpha = "{}".format(node.alpha)
        write_to_file(topo_file, "%s %s %s %s 0\n" % (node.op, name, ipt_name, str_alpha))
    except Exception as e:
        write_to_file(topo_file, "%s %s %s 0 0\n" % (node.op, name, ipt_name))


def dump_fc(node, name, input_name, first_fc, input_shape, fc_weights, have_bias, fc_bias, output_size, topo_file, weights_file, data_format, x_of_nChwxc):
    fc_weights = np.transpose(fc_weights, (1, 0))
    fc_weights_caffe = fc_weights
    fc_weights_mkldnn = fc_weights
    o_shape, i_shape = fc_weights_mkldnn.shape[0], fc_weights_mkldnn.shape[1]
    if first_fc:
        height, width, channel = 0, 0, 0
        if data_format == 'NHWC':
            height, width, channel = input_shape[1], input_shape[2], input_shape[3]
            fc_weights_mkldnn = fc_weights_mkldnn.reshape(o_shape, height, width, channel/x_of_nChwxc, x_of_nChwxc) # nhwc
            fc_weights_mkldnn = np.transpose(fc_weights_mkldnn, (0, 3, 1, 2, 4))
            fc_weights_caffe = fc_weights_caffe.reshape(o_shape, height, width, channel)
            fc_weights_caffe = np.transpose(fc_weights_caffe, (0, 3, 1, 2))
            fc_weights_caffe = fc_weights_caffe.reshape(o_shape, i_shape)
        else:
            height, width, channel = input_shape[2], input_shape[3], input_shape[1]
            fc_weights_mkldnn = fc_weights_mkldnn.reshape(o_shape, channel/x_of_nChwxc, x_of_nChwxc, height, width) # nhwc
            fc_weights_mkldnn = np.transpose(fc_weights_mkldnn, (0, 1, 3, 4, 2))
        fc_weights_mkldnn = fc_weights_mkldnn.reshape(o_shape, i_shape)
        #print 'o_shape[%d], i_shape[%d], height[%d], width[%d], channel[%d]' % (o_shape, i_shape, height, width, channel)
    if not have_bias:
        fc_bias = np.zeros(shape = fc_weights.shape[0])

    write_to_file(topo_file, "# op origin_name out_channel in_channel input_name\n")
    ipt_name = " ".join(input_name) if isinstance(input_name, list) else input_name
    write_to_file(topo_file, "%s %s %s %s %s\n" % (node.op, name, o_shape, i_shape, ipt_name))
    fc_weights_mkldnn.tofile(weights_file)
    g_weights.append(fc_weights_caffe)
    fc_bias.tofile(weights_file)
    g_weights.append(fc_bias)


# Dump convolution weights and bias to file
def dump_convolution(node, name, input_name, conv_weights, have_bias, conv_bias, input_shape, output_shape, pad, topo_file, weights_file):
    logger.debug("dump_convolution")
    logger.debug("node name:%s conv_weights_shape: %s" %(node.name, str(conv_weights.shape)))
    data_format = decode_string(node.attr.get('data_format').s)
    padding = decode_string(node.attr.get('padding').s)
    strides = node.attr.get('strides').list.i
    # Kernel size is got from the weights shape
    ksize_h = conv_weights.shape[0]
    ksize_w = conv_weights.shape[1]

    if len(input_name) == 0: input_name.append('Placeholder')

    if data_format == 'NHWC':
        stride_h = strides[1]
        stride_w = strides[2]
        in_channel = input_shape[3]
        out_channel = output_shape[3]
    else:
        stride_h = strides[2]
        stride_w = strides[3]
        in_channel = input_shape[1]
        out_channel = output_shape[1]

    if len(pad) > 4:
        pad_l, pad_r, pad_t, pad_b = pad[4], pad[5], pad[6], pad[7]
    else:
        pad_l, pad_r, pad_t, pad_b = get_pad_value(node, padding, data_format, ksize_h, ksize_w, stride_h, stride_w, input_shape, output_shape)

    if node.op in ['Conv2D', 'Conv2DBackpropInput']:
        write_to_file(topo_file, 
            "# op origin_name with_bias out_channel pad_l pad_r pad_t pad_b ksize_h ksize_w stride_h stride_w input_name\n")
        write_to_file(topo_file, "%s %s %s " % (node.op, name, have_bias))

        ipt_name = " ".join(input_name) if isinstance(
            input_name, list) else input_name
        write_to_file(topo_file, "%d %d %d %d %d %d %d %d %d %s\n" % (
            out_channel, pad_l, pad_r, pad_t, pad_b, ksize_h, ksize_w, stride_h, stride_w, ipt_name))

        dump_conv_weights(node.op, conv_weights, have_bias, conv_bias, weights_file)
        logger.debug("pad: %s" % (str(pad)))

    elif node.op == 'DepthwiseConv2dNative':
        write_to_file(topo_file, 
            "# op origin_name with_bias out_channel pad_l pad_r pad_t pad_b ksize_h ksize_w stride_h stride_w group input_name\n")
        write_to_file(topo_file, "%s %s %s " % (node.op, name, have_bias))
        ipt_name = " ".join(input_name) if isinstance(
            input_name, list) else input_name
        write_to_file(topo_file, "%d %d %d %d %d %d %d %d %d %d %s\n" % (
            out_channel, pad_l, pad_r, pad_t, pad_b, ksize_h, ksize_w, stride_h, stride_w, in_channel, ipt_name))
        dump_depthwiseconv_weights(
            conv_weights, have_bias, conv_bias, weights_file)
        logger.debug("pad: %s" % (str(pad)))


def dump_conv_weights(conv_op, conv_weights, have_bias, conv_bias, weights_file):
    if conv_op == 'Conv2DBackpropInput': 
        conv_weights = np.transpose(conv_weights, (2, 3, 0, 1))
    else:
        conv_weights = np.transpose(conv_weights, (3, 2, 0, 1))
    conv_weights.tofile(weights_file)
    g_weights.append(conv_weights)
    if have_bias:
        conv_bias.tofile(weights_file)
        g_weights.append(conv_bias)

# Dump depthwise conv weights and bias to file
def dump_depthwiseconv_weights(conv_weights, have_bias, conv_bias, weights_file):
    #[kernel_h] [kernel_w] [input_channle] [multiplier] -> [kernel_h] [kernel_w] [input_channle*multiplier] [1]
    kernel_h = conv_weights.shape[0]
    kernel_w = conv_weights.shape[1]
    input_channel = conv_weights.shape[2]
    multiplier = conv_weights.shape[3]
    conv_weights = conv_weights.reshape(
        kernel_h, kernel_w, input_channel * multiplier, 1)

    conv_weights = np.transpose(conv_weights, (2, 3, 0, 1))

    conv_weights.tofile(weights_file)
    g_weights.append(conv_weights)
    if have_bias:
        conv_bias.tofile(weights_file)
        g_weights.append(conv_bias)


# Refer to tensorflow.org/api_guides/python/nn#Convolution
def get_pad_value(node, padding, data_format, ksize_h, ksize_w, stride_h, stride_w, input_shape, output_shape):
    if node.op == "Conv2DBackpropInput": 
        tmp_shape = input_shape
        input_shape = output_shape
        output_shape = input_shape

    if data_format == 'NHWC':
        in_height = int(input_shape[1])
        in_width = int(input_shape[2])
        out_height = int(output_shape[1])
        out_width = int(output_shape[2])
    else:
        in_height = int(input_shape[2])
        in_width = int(input_shape[3])
        out_height = int(output_shape[2])
        out_width = int(output_shape[3])

    if padding == 'VALID':
        pad_left = 0
        pad_top = 0
        pad_right = max((out_width - 1) * stride_w + ksize_w - in_width, 0)
        pad_bottom = max((out_height - 1) * stride_h + ksize_h - in_height, 0)

    elif padding == 'SAME':
        if (in_height % stride_h == 0):
            pad_along_height = max(ksize_h - stride_h, 0)
        else:
            pad_along_height = max(ksize_h - (in_height % stride_h), 0)

        if (in_width % stride_w == 0):
            pad_along_width = max(ksize_w - stride_w, 0)
        else:
            pad_along_width = max(ksize_w - (in_width % stride_w), 0)

        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

    else:
        print("Unsupported padding(%s) for %s op" % (padding, node.op))
        exit(-1)

    return pad_left, pad_right, pad_top, pad_bottom


def dump_mean(node, name, input_name, k_h, k_w, s_h, s_w, topo_file):
    # eqaul to ave pooling 
    ipt_name = " ".join(input_name) if isinstance(
        input_name, list) else input_name
    write_to_file(topo_file, 
        "# op origin_name pad_l pad_r pad_t pad_b ksize_h ksize_w stride_h stride_w input_name\n")
    write_to_file(topo_file, "AvgPool %s 0 0 0 0 %s %s %s %s %s\n" % (name, k_h, k_w, s_h, s_w, ipt_name))


def dump_pool(node, name, input_name, input_shape, output_shape, topo_file):
    data_format = decode_string(node.attr.get('data_format').s)
    ksize = node.attr.get('ksize').list.i
    padding = decode_string(node.attr.get('padding').s)
    strides = node.attr.get('strides').list.i

    if data_format == 'NHWC':
        stride_h = strides[1]
        stride_w = strides[2]
        ksize_h = ksize[1]
        ksize_w = ksize[2]
    else:
        stride_h = strides[2]
        stride_w = strides[3]
        ksize_h = ksize[2]
        ksize_w = ksize[3]

    write_to_file(topo_file, 
        "# op origin_name pad_l pad_r pad_t pad_b ksize_h ksize_w stride_h stride_w input_name\n")

    pad_l, pad_r, pad_t, pad_b = get_pad_value(node, padding, data_format, ksize_h,
                        ksize_w, stride_h, stride_w, input_shape, output_shape)
    ipt_name = " ".join(input_name) if isinstance(
        input_name, list) else input_name
    write_to_file(topo_file, "%s %s %d %d %d %d %d %d %d %d %s\n" % (node.op, name, pad_l, pad_r, pad_t, pad_b, ksize_h, ksize_w, stride_h, stride_w, ipt_name))


def dump_add(node, topo_file, input_list, idx):
    write_to_file(topo_file, "# op origin_name input_name\n")
    write_to_file(topo_file, "Add %s %s %s\n" %
                    ('add' + str(idx), input_list[0], input_list[1]))


def dump_batchnorm(node, name, input_name, mean, variance, e, alpha, beta, topo_file, weights_file):
    write_to_file(topo_file, "# op origin_name epsilon input_name\n")
    write_to_file(topo_file, "BatchNorm %s %s %s\n" % (name, e, input_name[0]))

    mean.tofile(weights_file)
    variance.tofile(weights_file)
    alpha.tofile(weights_file)
    beta.tofile(weights_file)

    g_weights.append(mean)
    g_weights.append(variance)
    g_weights.append(1.0)
    g_weights.append(alpha)
    g_weights.append(beta)


def dump_extract_image_patches(node, name, input_name, input_shape, output_shape, topo_file):
    ksize = node.attr.get('ksizes').list.i
    padding = decode_string(node.attr.get('padding').s)
    strides = node.attr.get('strides').list.i
    rates = node.attr.get('rates').list.i

    # Strides
    stride_h, stride_w = strides[1], strides[2]
    ksize_h, ksize_w = ksize[1], ksize[2]
    rate_h, rate_w = rates[1], rates[2]

    write_to_file(topo_file, 
        "# op origin_name pad_l pad_r pad_t pad_b ksize_h ksize_w stride_h stride_w rate_h rate_w input_name\n")

    pad_l, pad_r, pad_t, pad_b = get_pad_value(node, padding, "NHWC", ksize_h,
                        ksize_w, stride_h, stride_w, input_shape, output_shape)
    ipt_name = " ".join(input_name) if isinstance(
        input_name, list) else input_name
    write_to_file(topo_file, "%s %s %d %d %d %d %d %d %d %d %d %d %s\n" % (node.op, name, pad_l, pad_r, pad_t, pad_b, ksize_h, ksize_w, stride_h, stride_w, rate_h, rate_w, ipt_name))


def dump_resize_bilinear(node, name, out_h, out_w, align_corners, input_name, topo_file):
    write_to_file(topo_file, 
        "# op origin_name out_height out_width align_corners input_name\n")

    ipt_name = " ".join(input_name) if isinstance(
        input_name, list) else input_name
    write_to_file(topo_file, "%s %s %s %s %s %s\n" % (node.op, name, out_h, out_w, align_corners, ipt_name))


if __name__ == "__main__":
    logger.info("utils")
