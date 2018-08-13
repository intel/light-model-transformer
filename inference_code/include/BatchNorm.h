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
#ifndef __BATCHNORM_H
#define __BATCHNORM_H
#include <iostream>
#include "mkldnn.hpp"
#include "Format.h"
#include "Relu.h"

using namespace std;
using namespace mkldnn;

class BatchNorm {
public:
  BatchNorm(const std::string &_name, 
          int _in_channels, int _in_h, int _in_w, 
          float epsilon,
          std::string _with_type = "relu",
          const std::string &_quantize_type = "fp32",
          float _scale_out = 1.0): // int8 or fp32
          name(_name), with_type(_with_type), quantize_type(_quantize_type), scale_out(_scale_out) {
    this->in_h = _in_h;
    this->in_w = _in_w;
    this->in_channels = _in_channels;
    this->epsilon = epsilon;
    
    p_weights = NULL;
    p_mean = NULL;
    p_variance = NULL;
    bnrm_prim_desc = NULL;
    bn_fd = NULL;
    p_src_memory = NULL;
    p_dst_memory = NULL;
  }

  ~BatchNorm() {
    delete p_weights;
    delete p_mean;
    delete p_variance;
    delete bnrm_prim_desc;
    delete bn_fd;
    delete p_src_memory;
    delete p_dst_memory;
  }

  // Return the destination/top memory
  memory *Init(int batch, engine *engine, memory &bottom, std::vector<primitive> &net,
                float* weights, float* mean, float* variance, std::string before_fmt) {
    fmt = before_fmt;
    std::cout << "[" << in_h << "*" << in_w << "*" << in_channels << "]\t" << name.c_str() << "(" << before_fmt << ")";
    this->cpu_engine = engine;

    int fmt_index = GetFormatIndex(before_fmt);
    memory::format format = formats[fmt_index];
    memory::dims src_tz = {batch, in_channels, in_h, in_w};
    auto conv_src_md = memory::desc({src_tz}, memory::data_type::f32, format);

    // batch_normalization_flag:
    //    use_global_stats, use_scale_shift, omit_stats, fuse_bn_relu
    unsigned flags = use_global_stats | use_scale_shift;
    auto bnrm_desc = batch_normalization_forward::desc(prop_kind::forward, conv_src_md, this->epsilon, flags);
    this->bnrm_prim_desc = new batch_normalization_forward::primitive_desc(bnrm_desc, *cpu_engine);

    memory::dims weights_tz = {in_channels};
    memory::dims double_weights_tz = {2, in_channels};
    p_weights = new memory({{{double_weights_tz}, memory::data_type::f32, memory::format::nc}, *cpu_engine}, weights);
    p_mean = new memory({{{weights_tz}, memory::data_type::f32, memory::format::x}, *cpu_engine}, mean);
    p_variance = new memory({{{weights_tz}, memory::data_type::f32, memory::format::x}, *cpu_engine}, variance);

    p_src_memory = new memory(bottom);
    p_dst_memory = new memory(bnrm_prim_desc->dst_primitive_desc());

    this->bn_fd = new batch_normalization_forward(*bnrm_prim_desc, *p_src_memory,
                            (const primitive::at)*p_mean, (const primitive::at)*p_variance, 
                            (const primitive::at)*p_weights, *p_dst_memory);
    net.push_back(*bn_fd);


    if (with_type == "Relu") {
        printf(" [with relu].\n");
        relu_dst_memory = relu(cpu_engine, bnrm_prim_desc, p_dst_memory, 0.0, 0.0, net);
        
        return relu_dst_memory;
    } else 
        printf("\n");
        return p_dst_memory;
  }

public:
  int getInputChannels() { return in_channels; }
  int getInputHeight() { return in_h; }
  int getInputWidth() { return in_w; }
  int getOutputChannels() { return in_channels; }
  int getOutputHeight() { return in_h; }
  int getOutputWidth() { return in_w; }
  std::string getQuantizeType() { return quantize_type; }
  std::string getFormat() { return fmt; }
  float getScaleOut() { return scale_out; }
  batch_normalization_forward::primitive_desc getPrim_desc() { return *bnrm_prim_desc; }

private:
  std::string name;
  int in_h, in_w, in_channels;
  float epsilon;
  
  string pool_type; // add

  std::string quantize_type;
  float scale_out;
  float alpha, beta;

  engine *cpu_engine;
  std::string fmt;

  memory *p_weights;
  memory *p_mean;
  memory *p_mean_memory;
  memory *p_variance;
  memory *p_src_memory;
  memory *p_dst_memory;
  batch_normalization_forward::primitive_desc *bnrm_prim_desc;
  primitive *bn_fd;

  std::string with_type;
  memory *relu_dst_memory;
  primitive *relu_fd;
};

#endif
