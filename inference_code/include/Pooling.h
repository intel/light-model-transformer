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
#ifndef __POOLING_H
#define __POOLING_H
#include <iostream>
#include "mkldnn.hpp"
#include "Format.h"

using namespace std;
using namespace mkldnn;

class Pooling {
public:
  Pooling(const std::string &_name, 
          int in_channels, 
          int in_h, int in_w, 
          int kernel_h, int kernel_w, 
          int pad_l = 0, int pad_r = 0, 
          int pad_t = 0, int pad_b = 0, 
          int stride_h = 1, int stride_w = 1,
          string pool_type = "Max",
          const std::string &_quantize_type = "fp32",
          float _scale_out = 1.0): // int8 or fp32
          name(_name), quantize_type(_quantize_type), scale_out(_scale_out) {
    this->out_channels = in_channels;
    this->in_channels = in_channels;
    this->in_h = in_h;
    this->in_w = in_w;
    this->kernel_h = kernel_h;
    this->kernel_w = kernel_w;
    if (pad_l == -1) {
        get_pad(in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, &this->pad_l, &this->pad_r, &this->pad_t, &this->pad_b);
    } else {
        this->pad_l = pad_l;
        this->pad_r = pad_r;
        this->pad_t = pad_t;
        this->pad_b = pad_b;
    }
    this->stride_h = stride_h;
    this->stride_w = stride_w;
    this->pool_type = pool_type; // add

    this->out_h = (in_h + this->pad_t + this->pad_b - kernel_h) / stride_h + 1;
    this->out_w = (in_w + this->pad_l + this->pad_r - kernel_w) / stride_w + 1;

    if ( this->out_h == 0 || this->out_w == 0 ) {
        std::cout << std::endl;
        std::cout << "  ERROR[Pool]:" << std::endl;
        std::cout << "      The out height or width of Layer [" << name;
        std::cout << "] is 0, please check the size of input picture!" << std::endl; 
        std::cout << std::endl;
        exit(-1);
    }

    this->fmt = "";

    p_src_memory = NULL;
    p_indices_memory = NULL;
    p_dst_memory = NULL;
    p_prim_desc = NULL;
    pool_fd = NULL;
  }

  ~Pooling() {
    delete p_src_memory;
    delete p_indices_memory;
    delete p_dst_memory;
    delete p_prim_desc;
    delete pool_fd;
  }

  // Return the destination/top memory
  memory *Init(int batch, engine *engine, memory &bottom, std::vector<primitive> &net) {
    this->cpu_engine = engine;

    memory::dims pool_src_tz = {batch, in_channels, in_h, in_w};
    memory::dims pool_dst_tz = {batch, out_channels, out_h, out_w};
    memory::dims pool_kernel = {kernel_h, kernel_w};
    memory::dims pool_strides = {stride_h, stride_w};
    auto pool_padding_l = {pad_t, pad_l};
    auto pool_padding_r = {pad_b, pad_r};

    memory::format mfmt_any = memory::format::any;
    memory::data_type bottom_dt = quantize_type == "int8" ? memory::data_type::u8 : memory::data_type::f32;
    memory::data_type top_dt = quantize_type == "int8" ? memory::data_type::u8 : memory::data_type::f32;

    auto pool_src_md = memory::desc({pool_src_tz}, bottom_dt, mfmt_any);
    auto pool_dst_md = memory::desc({pool_dst_tz}, top_dt, mfmt_any);

    // create a pooling
    auto pool_desc = pooling_forward::desc(prop_kind::forward, pool_type=="Max" ? pooling_max : pooling_avg,
                                    bottom.get_primitive_desc().desc(), pool_dst_md, pool_strides,
                                    pool_kernel, pool_padding_l, pool_padding_r, padding_kind::zero);
    this->p_prim_desc = new pooling_forward::primitive_desc(pool_desc, *cpu_engine);

    this->p_dst_memory = new memory(p_prim_desc->dst_primitive_desc());

    // Seems pooling does not need to reorder the src
    if ( pool_type=="Max" ) {
        // create pooling indices memory from pooling primitive descriptor
        this->p_indices_memory = new memory(p_prim_desc->workspace_primitive_desc());
        this->pool_fd = new pooling_forward(*p_prim_desc, bottom,
                                            *p_dst_memory, *p_indices_memory);
    }
    else {
        this->pool_fd = new pooling_forward(*p_prim_desc, bottom, *p_dst_memory);
    }
 
    net.push_back(*pool_fd);
    fmt = GetOutputFormat(cpu_engine, p_prim_desc, top_dt, pool_dst_tz);
	std::cout << "[" << out_h << "*" << out_w << "*" << out_channels << "]\t" << name << "(" << fmt << ")" << std::endl;

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
  std::string getFormat() { return fmt; }
  std::string getQuantizeType() { return quantize_type; }
  float getScaleOut() { return scale_out; }

private:
  std::string name;
  int out_channels;
  int in_channels;
  int in_h;
  int in_w;
  int kernel_h;
  int kernel_w;
  int pad_l;
  int pad_r;
  int pad_t;
  int pad_b;
  int stride_h; 
  int stride_w;
  int out_h;
  int out_w;
  string pool_type; // add

  std::string quantize_type;
  float scale_out;

  engine *cpu_engine;
  std::string fmt;

  memory *p_src_memory; // used for reorder
  memory *p_indices_memory;
  memory *p_dst_memory;
  pooling_forward::primitive_desc *p_prim_desc;
  primitive *pool_fd;
};

#endif
