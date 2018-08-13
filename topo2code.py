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
# -*- coding: utf-8 -*-
import sys
import cv2
import os
import struct
import argparse

from string import Template


def get_int8_topo_str(graph_dict):
    printf("unimplemented.")
    exit()


def get_delete_outs_str(graph_dict):
    _str = ""
    for key in range(len(graph_dict)):
        if key == 0: continue
        alias_name = graph_dict[key]["alias_name"]
        _str += "{0}delete {1}_out; {1}_out = NULL;\n".format(" "*8, alias_name)

    return _str


def get_delete_vars_str(graph_dict):
    delete_vars_str = ""
    for key in range(len(graph_dict)):
        if key == 0: continue
        if graph_dict[key]["op"] in ["Conv2D", "DepthwiseConv2dNative"]:
            alias_name = graph_dict[key]["alias_name"]
            delete_vars_str += "{0}delete[] {1}_w;\n".format(" "*8, alias_name)
            delete_vars_str += "{0}delete[] {1}_b;\n".format(" "*8, alias_name)
            
    return delete_vars_str


def get_create_outs_str(graph_dict, quantize):
    _str = ""
    __str = ""
    for key in range(len(graph_dict)):
        if key == 0: continue

        op, origin_name, alias_name = \
                            graph_dict[key]["op"], \
                            graph_dict[key]["origin_name"], \
                            graph_dict[key]["alias_name"]
        if quantize == "fp32" or op in ["Softmax", "MatMul"]:
            _str += "{0}printf(\"    --< {1} >--\\n\");\n".format(" "*8, origin_name)
            _str += "{0}printf(\"    >> [{1}".format(" "*8, "%f "*10)
            _str = _str[:-1] + "]\\n\\n\", "
            for i in range(10): _str += "p{0}[{1}], ".format(alias_name, i)
            _str = _str[:-2] + ");\n"

    if __str != "": _str += "\n{0}".format(__str)

    return _str


def get_init_outs_str(graph_dict, quantize):
    _str = ""
    __str = ""
    for key in range(len(graph_dict)):
        if key == 0: continue

        op, origin_name, alias_name = \
                            graph_dict[key]["op"], \
                            graph_dict[key]["origin_name"], \
                            graph_dict[key]["alias_name"]
        if quantize == "fp32" or op in ["Softmax", "MatMul"]:
            _str += "{0}float *p{1} = (float*){1}_out->get_data_handle();\n".format(" "*8, alias_name)

    if __str != "": _str += "\n{0}".format(__str)

    return _str


def get_define_outs_str(graph_dict):
    _str = ""
    for key in range(len(graph_dict)):
        if key == 0: continue

        alias_name = graph_dict[key]["alias_name"]
        _str += "    memory* {0}_out;\n".format(alias_name)

    return _str


def get_init_net_str(graph_dict):
    _str = ""
    for key in range(len(graph_dict)):
        if key == 0: continue

        op = graph_dict[key]["op"]
        alias_name = graph_dict[key]["alias_name"]
        if op == "ConcatV2":
            _index = graph_dict[key]["alias_name"].split("ConcatV2_")[1]
            _str += "{0}std::vector<std::string> bm_fmts_{1};\n".format(" "*8, _index)
            _str += "{0}vector<memory> bottoms_{1};\n".format(" "*8, _index)

            input_len = len(graph_dict[key]["input_name"])
            for i in range(input_len):
                _str += "{0}bm_fmts_{1}.push_back({2}->getFormat());\n".format(" "*8, _index, graph_dict[key]["input_name"][i])
                _str += "{0}bottoms_{1}.push_back(*{2}_out);\n".format(" "*8, _index, graph_dict[key]["input_name"][i])
            _str += "{0}{1}_out = {1}->Init(batch_size, &cpu_engine, bm_fmts_{2}, bottoms_{2}, net);\n".format(" "*8, alias_name, _index)

        else:
            if graph_dict[key]["input_name"][0] == "Placeholder":
            #if key == 1:
                input_str = "*src_memory"
            elif op == "Add":
                input_str = "{0}->getFormat(), {1}->getFormat(), *{0}_out, *{1}_out".format( \
                                graph_dict[key]["input_name"][0], graph_dict[key]["input_name"][1])
            else:
                input_str = "*{0}_out".format(graph_dict[key]["input_name"][0])

            _str += "{0}{1}_out = {1}->Init(batch_size, &cpu_engine, {2}, net".format(" "*8, alias_name, input_str)
            if op in ["Conv2D", "DepthwiseConv2dNative", "MatMul"]:
                _str += ", {0}_w, {0}_b".format(alias_name)
            elif op == "BatchNorm":
                _str += ", {0}_weights, {0}_mean, {0}_variance, {1}->getFormat()".format(alias_name, graph_dict[key]["input_name"][0])
            _str += ");\n"
            
    return _str


def get_read_vars_str(graph_dict):
    _str = ""

    for key in range(len(graph_dict)):
        if key == 0: continue

        op = graph_dict[key]["op"]
        alias_name = graph_dict[key]["alias_name"]
        if op not in ["Conv2D", "DepthwiseConv2dNative", "MatMul", "BatchNorm"]: continue

        _str += "\n        // origin_layer: {0}\n".format(graph_dict[key]["origin_name"])
        if op in ["Conv2D", "DepthwiseConv2dNative"]:
            _str += "{0}fread({1}_w, sizeof(float),\n".format(" "*8, alias_name)
            _str += "{0}{1}->getKernelHeight()".format(" "*14, alias_name)
            _str += " * {0}->getKernelWidth()".format(alias_name)
            _str += " * {0}->getOutputChannels()".format(alias_name)
            if op == "Conv2D":
                _str += " * {0}->getInputChannels()".format(alias_name)
            _str += ", fp);\n"

            if graph_dict[key]["with_bias"] == "True":
                _str += "{0}fread({1}_b, sizeof(float), {1}->getOutputChannels(), fp);\n".format(" "*8, alias_name)
            else:
                _str += "{0}for(int i = 0; i < {1}->getOutputChannels(); i ++) {1}_b[i] = 0.0;\n".format(" "*8, alias_name)

        elif op == "MatMul":
            _str += "{0}fread({1}_w, sizeof(float), {1}->getInputChannels()".format(" "*8, alias_name)
            _str += " * {0}->getOutputChannels(), fp);\n".format(alias_name)
            _str += "{0}fread({1}_b, sizeof(float), {1}->getOutputChannels(), fp);\n".format(" "*8, alias_name)

        elif op == "BatchNorm":
            _str += "{0}fread({1}_mean, sizeof(float), {1}->getOutputChannels(), fp);\n".format(" "*8, alias_name)
            _str += "{0}fread({1}_variance, sizeof(float), {1}->getOutputChannels(), fp);\n".format(" "*8, alias_name)
            _str += "{0}fread({1}_weights, sizeof(float), {1}->getOutputChannels() * 2, fp);\n".format(" "*8, alias_name)

    return _str


def get_create_vars_str(graph_dict):
    _str = ""

    for key in range(len(graph_dict)):
        op = graph_dict[key]["op"]
        alias_name = graph_dict[key]["alias_name"]
        out_channel = graph_dict[key]["out_channel"]
        
        if op in ["Conv2D", "DepthwiseConv2dNative"]:
            ksize_h = graph_dict[key]["ksize_h"]
            ksize_w = graph_dict[key]["ksize_w"]
            in_channel = graph_dict[key]["in_channel"]
            _str += "float* Model::{0}_w = new float[{1} * {2} * {3} * {4}];\n".format(alias_name, ksize_h, ksize_w, out_channel, in_channel)
            _str += "float* Model::{0}_b = new float[{1}];\n".format(alias_name, out_channel)
        elif op == "MatMul":
            in_channel = graph_dict[key]["in_channel"]
            _str += "float* Model::{0}_w = new float[{1} * {2}];\n".format(alias_name, out_channel, in_channel)
            _str += "float* Model::{0}_b = new float[{1}];\n".format(alias_name, out_channel)
        elif op == "BatchNorm":
            _str += "float* Model::{0}_weights = new float[{1} * 2];\n".format(alias_name, out_channel)
            _str += "float* Model::{0}_mean = new float[{1}];\n".format(alias_name, out_channel)
            _str += "float* Model::{0}_variance = new float[{1}];\n".format(alias_name, out_channel)

    return _str


def get_define_vars_str(graph_dict):
    define_vars_str = ""
    for key in range(len(graph_dict)):
        if key == 0: continue

        op = graph_dict[key]["op"]
        alias_name = graph_dict[key]["alias_name"]
        if op in ["Conv2D", "DepthwiseConv2dNative", "MatMul"]:
            define_vars_str += "    static float *{0}_w, *{0}_b;\n".format(alias_name)
        elif op == "BatchNorm":
            define_vars_str += "    static float *{0}_weights, *{0}_mean, *{0}_variance;\n" .format(alias_name)

    return define_vars_str


def get_create_net_str(graph_dict, quantize):
    _str = ""
    for key in range(len(graph_dict)):
        if key == 0: continue

        if graph_dict[key]["input_name"][0] == "Placeholder":
            channel, height, width, quantize_type = \
                                "input_channel", \
                                "input_height", \
                                "input_width", \
                                "\"fp32\""
        else:
            input_name = graph_dict[key]["input_name"][0]
            channel, height, width, quantize_type = \
                                "{0}->getOutputChannels()".format(input_name), \
                                "{0}->getOutputHeight()".format(input_name), \
                                "{0}->getOutputWidth()".format(input_name), \
                                "{0}->getQuantizeType()".format(input_name)
            
        op = graph_dict[key]["op"]
        alias_name = graph_dict[key]["alias_name"]
        out_channel = graph_dict[key]["out_channel"]

        if op in ["Conv2D", "DepthwiseConv2dNative", "AvgPool", "MaxPool", "ExtractImagePatches"]:
            ksize_h, ksize_w = graph_dict[key]["ksize_h"], graph_dict[key]["ksize_w"]
            pad_l, pad_r = graph_dict[key]["pad_l"], graph_dict[key]["pad_r"]
            pad_t, pad_b = graph_dict[key]["pad_t"], graph_dict[key]["pad_b"]
            stride_h, stride_w = graph_dict[key]["stride_h"], graph_dict[key]["stride_w"]

        if op in ["Conv2D", "DepthwiseConv2dNative", "Add", "MatMul", "BatchNorm"]:
            with_type = graph_dict[key]["with_type"]
            if op in ["Conv2D", "DepthwiseConv2dNative", "MatMul"]:
                alpha, beta = graph_dict[key]["alpha"], graph_dict[key]["beta"]

        # Combine string
        _str += "\n        // origin_layer: {0}\n".format(graph_dict[key]["origin_name"])
        if op in ["Conv2D", "DepthwiseConv2dNative"]:
            _str += "{0}{1} = new Convolution(\"{1}\", {2},\n".format(" "*8, alias_name, out_channel)
            _str += "{0}{1}, {2}, {3},\n".format(" "*24, channel, height, width)
            _str += "{0}{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8},\n".format( \
                                    " "*24, ksize_h, ksize_w, \
                                    pad_l, pad_r, pad_t, pad_b, \
                                    stride_h, stride_w)
            _str += "{0}\"{1}\",\n".format(" "*24, op)
            _str += "{0}\"{1}\", {2}, {3},\n".format(" "*24, with_type, alpha, beta)
            _str += "{0}\"fp32\", 0.0, 0.0, 0.0);\n".format(" "*24)

        elif op in ["AvgPool", "MaxPool"]:
            _str += "{0}{1} = new Pooling(\"{1}\",\n".format(" "*8, alias_name)
            _str += "{0}{1}, {2}, {3},\n".format(" "*24, channel, height, width)
            _str += "{0}{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8},\n".format( \
                                    " "*24, ksize_h, ksize_w, \
                                    pad_l, pad_r, pad_t, pad_b, \
                                    stride_h, stride_w)
            _str += "{0}\"{1}\",\n".format(" "*24, op[:3])
            _str += "{0}{1});\n".format(" "*24, quantize_type)

        elif op == "Softmax":
            _str += "{0}{1} = new Softmax(\"{1}\",".format(" "*8, alias_name)
            if "MatMul" in graph_dict[key]["input_name"][0]:
                _str += " {0});\n".format(channel)
            else:
                _str += " {0} * {1} * {2});\n".format(channel, height, width)

        elif op == "Add":
            if quantize == "fp32":
                scale_in1, scale_in2 = "1.0", "1.0"
                quantize_type1, quantize_type2 = "\"fp32\"", "\"fp32\""
            else:
                scale_in1, scale_in2 = graph_dict[key]["scale_in1"], graph_dict[key]["scale_in2"]
                quantize_type1 = "{0}->getQuantizeType()".format(graph_dict[key]["input_name"][0])
                quantize_type2 = "{0}->getQuantizeType()".format(graph_dict[key]["input_name"][1])
            _str += "{0}{1} = new Sum(\"{1}\", \n".format(" "*8, alias_name, out_channel)
            _str += "{0}{1}, {2}, {3},\n".format(" "*24, channel, height, width)
            _str += "{0}\"{1}\",\n".format(" "*24, with_type)
            _str += "{0}{1}, {2}, {3}, {4});\n".format(" "*24, quantize_type1, quantize_type2, scale_in1, scale_in2)
            
        elif op == "MatMul":
            _str += "{0}{1} = new InnerProduct(\"{1}\", {2},\n".format(" "*8, alias_name, out_channel)
            if "MatMul" in graph_dict[key]["input_name"][0]:
                _str += "{0}{1}, 1, 1,\n".format(" "*24, channel)
            else:
                _str += "{0}{1}, {2}, {3},\n".format(" "*24, channel, height, width)
            _str += "{0}\"{1}\", {2}, {3},\n".format(" "*24, with_type, alpha, beta)
            _str += "{0}{1}, {2}->getScaleOut());\n".format(" "*24, quantize_type, input_name)

        elif op == "ConcatV2":
            concat_index = graph_dict[key]["alias_name"].split("ConcatV2_")[1]
            _str += "{0}vector<std::string> quantizes_{1};\n".format(" "*8, concat_index)
            _str += "{0}vector<float> scales_{1};\n".format(" "*8, concat_index)
            _str += "{0}vector<int> dims_{1}(3, -1);\n".format(" "*8, concat_index)
            _str += "{0}vector<vector<int> > in_dims_{1};\n".format(" "*8, concat_index)

            input_len = len(graph_dict[key]["input_name"])
            for i in range(input_len):
                input_name = graph_dict[key]["input_name"][i]
                _str += "{0}quantizes_{1}.push_back({2}->getQuantizeType()); ".format(" "*8, concat_index, input_name)
                if quantize == "fp32":
                    _str += "scales_{0}.push_back(1.0);\n".format(concat_index)
                else:
                    _str += "scales_{0}.push_back({1});\n".format(concat_index, graph_dict[key]["scale"][i])
                _str += "{0}dims_{1}[0] = {2}->getOutputChannels();".format(" "*8, concat_index, input_name)
                _str += " dims_{0}[1] = {1}->getOutputHeight();".format(concat_index, input_name)
                _str += " dims_{0}[2] = {1}->getOutputWidth();\n".format(concat_index, input_name)
                _str += "{0}in_dims_{1}.push_back(dims_{2});\n".format(" "*8, concat_index, concat_index)

            _str += "{0}{1} = new Concat(\"{1}\", \n".format(" "*8, alias_name)
            _str += "{0}in_dims_{1}, \'c\', quantizes_{1}, scales_{1});\n".format(" "*24, concat_index)

        elif op == "BatchNorm":
            epsilon = graph_dict[key]["epsilon"]
            _str += "{0}{1} = new BatchNorm(\"{1}\", \n".format(" "*8, alias_name)
            _str += "{0}{1}, {2}, {3}, {4}, \"{5}\");\n".format(" "*24, channel, height, width, epsilon, with_type)

        elif op == "ExtractImagePatches":
            rate_h, rate_w = graph_dict[key]["rate_h"], graph_dict[key]["out_weight"]
            _str += "{0}{1} = new ExtractImagePatches(\"{1}\", \n".format(" "*8, alias_name)
            _str += "{0}{1}, {2}, {3},\n".format(" "*24, channel, height, width)
            _str += "{0}{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8},\n".format( \
                                    " "*24, ksize_h, ksize_w, \
                                    pad_l, pad_r, pad_t, pad_b, \
                                    stride_h, stride_w)
            _str += "{0}{1}, {2},\n".format(" "*24, rate_h, rate_w)
            _str += "{0}{1});\n".format(" "*24, quantize_type)

        elif op == "ResizeBilinear":
            out_h, out_w = graph_dict[key]["out_height"], graph_dict[key]["out_width"]
            _str += "{0}{1} = new ResizeBilinear(\"{1}\", \n".format(" "*8, alias_name)
            _str += "{0}{1}, {2}, {3},\n".format(" "*24, channel, height, width)
            _str += "{0}{1}, {2},\n".format(" "*24, out_h, out_w)
            _str += "{0}{1});\n".format(" "*24, quantize_type)
            
    return _str


def get_define_net_str(graph_dict, quantize):
    op_2_mkldnn = {
        "Conv2D": "Convolution*",
        "DepthwiseConv2dNative": "Convolution*",
        "Conv2DBackpropInput": "Deconvolution*",
        "AvgPool": "Pooling*",
        "MaxPool": "Pooling*",
        "Mean": "Pooling*",
        "Softmax": "Softmax*",
        "Add": "Sum*",
        "MatMul": "InnerProduct*",
        "ConcatV2": "Concat*",
        "BatchNorm": "BatchNorm*",
        "ExtractImagePatches": "ExtractImagePatches*",
        "ResizeBilinear": "ResizeBilinear*",
        }

    define_net_str = ""
    define_net_str += "    // {0:<20} {1:<10} {2}\n".format("Type", "Name", "Origin_name")
    for key in range(len(graph_dict)):
        if key == 0: continue
        op = graph_dict[key]["op"]
        define_net_str += "    {0:<20} {1};".format(op_2_mkldnn[op], graph_dict[key]["alias_name"])
        define_net_str += "\t  // {0}\n".format(graph_dict[key]["origin_name"])
            
    return define_net_str


def topo_2_dict(topo_file):
    content = ""
    with open(topo_file, "r") as f:
        content = f.read()

    # Get content from topo.txt
    key = 0
    type_list, node_list = [], []
    graph_dict = {}
    dict_merge = {} # alias_name: input_alias_name
    dict_alias_key = {} # alias_name: key_of_graph
    dict_op_counter = {} # op_name: op_counter
    dict_origin_alias = {} # origin_layer_name: alias_layer_name
    for line in content.split("\n"):
        if len(line) == 0: continue

        if line[0] == "#": type_list = line.split(" ")[1:]
        else: node_list = line.split(" ")
        
        if len(type_list) > 0 and len(node_list) > 0:
            if node_list[0] in ["Relu", "Relu6"]:
                dict_merge[node_list[1]] = node_list[2]
                graph_dict[key-1]["with_type"] = node_list[0]
                graph_dict[key-1]["alpha"] = node_list[3]
                if node_list[0] == "Relu6":
                    graph_dict[key-1]["beta"] = "6"
                continue
            
            _op = node_list[0]
            if _op not in dict_op_counter.keys(): dict_op_counter[_op] = 1
            else: dict_op_counter[_op] += 1

            if _op == "Placeholder": alias_name = "Placeholder"
            else: alias_name = "{0}_{1}".format(_op, dict_op_counter[_op])
            dict_origin_alias[node_list[1]] = alias_name
            dict_alias_key[alias_name] = key

            graph_dict[key] = {}
            graph_dict[key]["alias_name"] = alias_name

            last_index = len(type_list) - 1
            for _index in range(last_index):
                type_ele, node_ele = type_list[_index], node_list[_index]
                graph_dict[key][type_ele] = node_ele

            type_ele = type_list[last_index]
            graph_dict[key][type_ele] = node_list[last_index:]

            if graph_dict[key]["op"] in ["Conv2D", "DepthwiseConv2dNative", "Add", "MatMul", "BatchNorm"]:
                graph_dict[key]["with_type"] = "none"
                graph_dict[key]["alpha"] = "0"
                graph_dict[key]["beta"] = "0"

            type_list, node_list = [], []
            key += 1

    # Modify input_name
    for key in graph_dict.keys():
        if graph_dict[key]["op"] == "Placeholder":
            graph_dict[key]["width"] = graph_dict[key]["width"][0]
            continue
        
        for _index in range(len(graph_dict[key]["input_name"])):
            _origin_name = graph_dict[key]["input_name"][_index]
            if _origin_name in dict_merge.keys():
                _origin_name = dict_merge[_origin_name]
            _alias_name = dict_origin_alias[_origin_name]
            graph_dict[key]["input_name"][_index] = _alias_name

    # Calculate output_channels
    for key in graph_dict.keys():
        op = graph_dict[key]["op"]

        if graph_dict[key]["op"] == "Placeholder": continue

        out_channel = 0
        if op in ["Conv2D", "DepthwiseConv2dNative", "MatMul"]:
            out_channel = graph_dict[key]["out_channel"]
            if op == "DepthwiseConv2dNative":
                out_channel = int(out_channel) / int(graph_dict[key]["group"])
        else:
            if op in ["Add", "ConcatV2"]:
                input_len = len(graph_dict[key]["input_name"])
                for i in range(input_len):
                    input_name = graph_dict[key]["input_name"][i]
                    input_key = dict_alias_key[input_name]
                    in_channel = int(graph_dict[input_key]["out_channel"])
                    if graph_dict[input_key]["op"] == "ExtractImagePatches":
                        in_channel = in_channel * int(graph_dict[input_key]["ksize_h"]) * int(graph_dict[input_key]["ksize_w"])
                    out_channel += int(in_channel)
            else:
                input_key = dict_alias_key[graph_dict[key]["input_name"][0]]
                out_channel = graph_dict[input_key]["out_channel"]

            graph_dict[key]["out_channel"] = out_channel

    # Get input_channels
    for key in graph_dict.keys():
        op = graph_dict[key]["op"]
        if op in ["Conv2D", "DepthwiseConv2dNative"]:
            input_key = dict_alias_key[graph_dict[key]["input_name"][0]]
            in_channel = int(graph_dict[input_key]["out_channel"])
            if graph_dict[input_key]["op"] == "ExtractImagePatches":
                in_channel = in_channel * int(graph_dict[input_key]["ksize_h"]) * int(graph_dict[input_key]["ksize_w"])

            graph_dict[key]["in_channel"] = in_channel


    return graph_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noinput", help="no input file", action="store_true")
    parser.add_argument("--input", default="inference_code/input/input_224.jpg", type=str, help="input file")
    parser.add_argument("--topo", default="save_model/topo.txt", type=str, help="topo file")
    parser.add_argument("--h", default=224, type=int, help="Need height of net input")
    parser.add_argument("--w", default=224, type=int, help="Need width of net input")
    parser.add_argument("--quantize", default="cpp_fp32", type=str, help="Quantize: topo, cpp_fp32 or cpp_int8")
    args = parser.parse_args()

    topo = args.topo
    if args.quantize != "cpp_int8":
        graph_dict = topo_2_dict(topo) 
    else:
        graph_dict = topo_2_dict("inference_code/topo/topo_int8.txt") 

    if args.quantize == "cpp_fp32":
        define_net_str = get_define_net_str(graph_dict, "fp32")
        create_net_str = get_create_net_str(graph_dict, "fp32")
        init_outs_str = get_init_outs_str(graph_dict, "fp32")
        create_outs_str = get_create_outs_str(graph_dict, "fp32")
        int8_topo_str = ""
    elif args.quantize == "topo":
        create_net_str = get_create_net_str(graph_dict, "fp32")
        init_outs_str = get_init_outs_str(graph_dict, "fp32")
        create_outs_str = ""
        int8_topo_str = get_int8_topo_str(graph_dict)
    else:
        create_net_str = get_create_net_str(graph_dict, "int8")
        init_outs_str = get_init_outs_str(graph_dict, "int8")
        create_outs_str = get_create_outs_str(graph_dict, "int8")
        int8_topo_str = ""

    define_vars_str = get_define_vars_str(graph_dict)
    define_outs_str = get_define_outs_str(graph_dict)
    read_vars_str = get_read_vars_str(graph_dict)
    init_net_str = get_init_net_str(graph_dict)
    create_vars_str = get_create_vars_str(graph_dict)
    delete_vars_str = get_delete_vars_str(graph_dict)
    delete_outs_str = get_delete_outs_str(graph_dict)

    last_key = len(graph_dict.keys()) - 1
    last_out_name_str = "%s_out" % graph_dict[last_key]["alias_name"]

    with open("cfg/Model.cfg", "r") as f:
        data = f.read()

    data = Template(data).safe_substitute( \
            height = graph_dict[0]["height"], \
            width = graph_dict[0]["width"], \
            channel = graph_dict[0]["out_channel"], \
            last_out_name = last_out_name_str, \
            define_net=define_net_str, \
            define_vars=define_vars_str, \
            define_outs=define_outs_str, \
            create_net=create_net_str, \
            init_net=init_net_str, \
            read_vars=read_vars_str, \
            init_outs=init_outs_str, \
            create_outs=create_outs_str, \
            create_vars=create_vars_str, \
            make_int8_topo=int8_topo_str, \
            delete_vars=delete_vars_str, \
            delete_outs = delete_outs_str)
    with open("inference_code/Model.hpp", "w") as f:
        f.write(data)

    print("[*] mission complete, you can run as below: \n\tcd inference_code \n\tsh build.sh")

