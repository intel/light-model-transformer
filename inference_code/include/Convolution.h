/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef __CONVOLUTION_H
#define __CONVOLUTION_H
#include <iostream>
#include "mkldnn.hpp"
#include "Format.h"
#include "Reorder.h"
#include "Pad.h"
#include "Relu.h"

using namespace mkldnn;

class Convolution {
public:
  Convolution(const std::string &_name, 
              int out_channels, int in_channels, 
              int in_h, int in_w, 
              int kernel_h, int kernel_w, 
              int l_pad = 0, int r_pad = 0, int t_pad = 0, int b_pad = 0, 
              int stride_h = 1, int stride_w = 1,
              const std::string &_type = "Conv2D", // 'Conv2D' or 'DepthwiseConv2dNative' convolution
              const std::string &_with_type = "none", // Relu or Relu6 or none
              float _alpha = 0, float _beta = 0, // alpha and beta of Relu or Relu6
              const std::string &_quantize_type = "fp32", // int8 or fp32
              float _scale_in = 1.0, float _scale_out = 1.0, float _scale_params = 1.0):
              name(_name), _type(_type),
              with_type(_with_type), alpha(_alpha), beta(_beta),
              quantize_type(_quantize_type),
              scale_in(_scale_in), scale_out(_scale_out), scale_params(_scale_params) {
    if ( with_type == "Relu6" && alpha == 0 && beta == 0 ) {
        printf("ERROR: Relu6 need parameter alpha and beta.\n");
        exit(-1);
    }
    if ( _type == "DepthwiseConv2dNative" ) 
        this->out_channels = in_channels * out_channels;
    else
        this->out_channels = out_channels;
    this->in_channels = in_channels;
    this->in_h = in_h;
    this->in_w = in_w;
    this->kernel_h = kernel_h;
    this->kernel_w = kernel_w;
    if (l_pad == -1) {
        get_pad(in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, &this->l_pad, &this->r_pad, &this->t_pad, &this->b_pad);
    } else {
        this->l_pad = l_pad;
        this->r_pad = r_pad;
        this->t_pad = t_pad;
        this->b_pad = b_pad;
    }
    this->stride_h = stride_h;
    this->stride_w = stride_w;
    this->out_h = (in_h + this->t_pad + this->b_pad - kernel_h) / stride_h + 1;
    this->out_w = (in_w + this->l_pad + this->r_pad - kernel_w) / stride_w + 1;
    this->fmt = "";
    //printf("%s - oh:%d, ow:%d, ic:%d, oc:%d\n", name.c_str(), out_h, out_w, this->in_channels, this->out_channels);

    if ( this->out_h == 0 || this->out_w == 0 ) {
        std::cout << std::endl;
        std::cout << "  ERROR[Conv]:" << std::endl;
        std::cout << "      The out height or width of Layer [" << name;
        std::cout << "] is 0, please check the size of input picture!" << std::endl; 
        std::cout << std::endl;
        exit(-1);
    }

    p_src_memory = NULL;
    p_weights_memory = NULL;
    p_bias_memory = NULL;
    p_dst_memory = NULL;
    p_dst_memory_u8 = NULL;
    user_dst_memory = NULL;
    p_prim_desc = NULL;
    _fd = NULL;
  }

  ~Convolution() {
    delete p_src_memory;
    delete p_weights_memory;
    delete p_bias_memory;
    delete p_dst_memory;
    delete p_dst_memory_u8;
    delete p_prim_desc;
    delete _fd;
  }

  memory *Init(int batch, engine *engine, memory &bottom, std::vector<primitive> &net, 
               float *weights, float *bias) {
    this->cpu_engine = engine;

    memory::dims _src_tz = {batch, in_channels, in_h, in_w};
    memory::dims _bias_tz = {out_channels};
    memory::dims _dst_tz = {batch, out_channels, out_h, out_w};
    memory::dims _strides = {stride_h, stride_w};
    memory::dims _weights_tz = {out_channels, in_channels, kernel_h, kernel_w};
    if ( _type == "DepthwiseConv2dNative" )
        _weights_tz = {in_channels, out_channels/in_channels, 1, kernel_h, kernel_w};

    auto _padding_l = {t_pad, l_pad};
    auto _padding_r = {b_pad, r_pad};
    
//    user_dst = new float[batch * out_channels * out_h * out_w];

    memory::format mfmt_any = memory::format::any;
    memory::data_type bottom_dt = quantize_type == "int8" ? memory::data_type::u8 : memory::data_type::f32;
    memory::data_type top_dt = quantize_type == "int8" ? memory::data_type::u8 : memory::data_type::f32;
    memory::data_type weights_dt = quantize_type == "int8" ? memory::data_type::s8 : memory::data_type::f32;
    memory::data_type bias_dt = quantize_type == "int8" ? memory::data_type::s8 : memory::data_type::f32;

    auto _src_md = memory::desc({_src_tz}, bottom_dt, mfmt_any);
    auto _dst_md = memory::desc({_dst_tz}, top_dt, mfmt_any);
    auto _weights_md = memory::desc({_weights_tz}, weights_dt, mfmt_any);
    auto _bias_md = memory::desc({_bias_tz}, bias_dt, mfmt_any);
        
    // create a convolution primitive description
    auto _desc = convolution_forward::desc(prop_kind::forward,
        convolution_direct, _src_md, _weights_md, _bias_md,
        _dst_md, _strides, _padding_l, _padding_r,
        padding_kind::zero);

    primitive_attr _attr;
    if ( quantize_type == "int8" ) {
        _scales = { scale_out/(scale_in*scale_params) };
        _attr.set_output_scales(0, _scales);
        _attr.set_int_output_round_mode(round_mode::round_nearest);
    }

    this->p_prim_desc = new
            convolution_forward::primitive_desc(_desc, _attr, *cpu_engine);

    fmt = GetOutputFormat(cpu_engine, p_prim_desc, top_dt, _dst_tz);
    std::cout << "[" << out_h << "*" << out_w << "*" << out_channels << "]\t" << name.c_str() << "(" << fmt << ")\t:";

    // reorder weights
    auto format = memory::format::oihw;
    if ( _type == "DepthwiseConv2dNative" ) format = memory::format::goihw;
        
    auto user_weights_memory = memory({{{_weights_tz},
        memory::data_type::f32, format}, *cpu_engine}, weights);
    if (memory::primitive_desc(p_prim_desc->weights_primitive_desc()) !=
            user_weights_memory.get_primitive_desc()) {
		std::cout << " Need to reorder the weights";
        p_weights_memory = new memory(p_prim_desc->weights_primitive_desc());
        Reorder reorder(scale_params);
        if ( quantize_type == "int8" ) {
            std::cout << "(s8)";
            p_weights_memory = reorder.Init(user_weights_memory, *p_weights_memory, true);
        } else {
            p_weights_memory = reorder.Init(user_weights_memory, *p_weights_memory, false);
        }
    } else {
        p_weights_memory = new memory({{{_weights_tz},
            memory::data_type::f32, format}, *cpu_engine}, weights);
    }

    // bias
    auto user_bias_memory = memory({{{_bias_tz}, 
        memory::data_type::f32, memory::format::x}, *cpu_engine}, bias);
    if ( quantize_type == "int8" ) {
		std::cout << ", bias(s8)";
        p_bias_memory = new memory(p_prim_desc->bias_primitive_desc());
        Reorder reorder(scale_in * scale_params);
        p_bias_memory = reorder.Init(user_bias_memory, *p_bias_memory, true);
    } else {
        p_bias_memory = new memory({{{_bias_tz},
                memory::data_type::f32, memory::format::x}, *cpu_engine}, bias);
    }

    if (memory::primitive_desc(p_prim_desc->src_primitive_desc()) !=
            bottom.get_primitive_desc()) {
        std::cout << ", input";
        p_src_memory = new memory(p_prim_desc->src_primitive_desc());
        if ( quantize_type == "int8" ) {
            std::cout << "(u8)";
            Reorder reorder(scale_in);
            p_src_memory = reorder.Init(bottom, *p_src_memory, net);
        } else {
            net.push_back(reorder(bottom, *p_src_memory));
        }
    } else {
        p_src_memory = new memory(bottom);
    }

    // Prepare dst/top memory
    p_dst_memory = new memory(p_prim_desc->dst_primitive_desc());

    _fd = new convolution_forward(*p_prim_desc, *p_src_memory,
               *p_weights_memory, *p_bias_memory, *p_dst_memory);

    net.push_back(*_fd);

    if ( with_type == "Relu6" ) { 
		printf(" - [with Relu6].\n");
        _dst_memory = relu(cpu_engine, p_prim_desc, p_dst_memory, 6.0, beta, net, with_type);

        return _dst_memory;
    } else if ( with_type == "Relu" ) { 
		printf(" - [with Relu].\n");
        _dst_memory = relu(cpu_engine, p_prim_desc, p_dst_memory, alpha, beta, net, with_type);

        return _dst_memory;
    } else {
		printf("\n");
	}

    return p_dst_memory;
  }

public:
  int getInputChannels() { return in_channels; }
  int getInputHeight() { return in_h; }
  int getInputWidth() { return in_w; }
  int getOutputChannels() { return out_channels; }
  int getOutputHeight() { return out_h; }
  int getOutputWidth() { return out_w; }
  int getKernelHeight() { return kernel_h; }
  int getKernelWidth() { return kernel_w; }
  float getScaleOut() { return scale_out; };
  std::string getFormat() { return fmt; }
  std::string getQuantizeType() { return quantize_type; }

private:
  std::string name;
  std::string _type;
  int out_channels;
  int in_channels;
  int in_h;
  int in_w;
  int kernel_h;
  int kernel_w;
  int l_pad;
  int r_pad;
  int t_pad;
  int b_pad;
  int stride_h; 
  int stride_w;
  int out_h;
  int out_w;
  float alpha, beta;

  std::string quantize_type;
  std::vector<float> _scales, in_scales;
  float scale_in, scale_out, scale_params;

  std::string with_type;

  engine *cpu_engine;
  std::string fmt;

  memory *p_src_memory; // used for reorder
  memory *p_weights_memory;
  memory *p_bias_memory;
  memory *p_dst_memory;

  memory *p_dst_memory_u8;
  memory *user_dst_memory;
  float *user_dst;

  memory *_dst_memory;
  convolution_forward::primitive_desc *p_prim_desc;
  reorder::primitive_desc *p_reorder_desc;
  primitive *_fd;
  primitive *relu6_fd;
};

#endif
