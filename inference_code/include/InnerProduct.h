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
#ifndef __INNERPRODUCT_H
#define __INNERPRODUCT_H
#include <iostream>
#include "mkldnn.hpp"
#include "Reorder.h"
#include "Format.h"
#include "Relu.h"

using namespace mkldnn;

class InnerProduct {
public:
  InnerProduct(const std::string &_name, 
               int _out_channels, int _in_channel, 
               int _in_height, int _in_width,
               std::string _with_type = "none",
               float _min=0,
               float _max=0,
               const std::string &_quantize_type = "fp32", float _scale_in = 1.0) :
               name(_name), with_type(_with_type), min(_min), max(_max), scale_in(_scale_in) {
    if ( with_type == "Relu6" && min == 0 && max == 0 ) {
        printf("ERROR: Relu6 need parameter min and max.\n");
        exit(-1);
    }
    this->out_channels = _out_channels;
    this->in_channel = _in_channel;
    this->in_height = _in_height;
    this->in_width = _in_width;

    this->in_channels = in_channel * in_height * in_width;

    this->quantize_type = _quantize_type;

    ip_src_memory = NULL; 
    ip_weights_memory = NULL;
    ip_bias_memory = NULL;
    ip_dst_memory = NULL;
    ip_prim_desc = NULL;
    ip_fd = NULL;
  }

  ~InnerProduct() {
    delete ip_src_memory;
    delete ip_weights_memory;
    delete ip_bias_memory;
    delete ip_dst_memory;
    delete ip_prim_desc;
    delete ip_fd;
  }

  memory *Init(int batch, engine *engine, memory &bottom, std::vector<primitive> &net, 
               float *weights, float *bias) {
    this->cpu_engine = engine;

    memory::dims ip_src_tz = {batch, in_channels};
    memory::dims ip_weights_tz = {out_channels, in_channels};
    memory::dims ip_bias_tz = {out_channels};
    memory::dims ip_dst_tz = {batch, out_channels};

    auto ip_src_md = memory::desc({ip_src_tz}, memory::data_type::f32, memory::format::any);
    auto ip_bias_md = memory::desc({ip_bias_tz}, memory::data_type::f32, memory::format::any);
    auto ip_weights_md = memory::desc({ip_weights_tz}, memory::data_type::f32, memory::format::any);
    auto ip_dst_md = memory::desc({ip_dst_tz}, memory::data_type::f32, memory::format::any);
        
    // Create a inner product primitive description
    auto ip_desc = inner_product_forward::desc(prop_kind::forward_inference,
        ip_src_md, ip_weights_md, ip_bias_md, ip_dst_md);
    this->ip_prim_desc = new
        inner_product_forward::primitive_desc(ip_desc, *cpu_engine);

    memory::data_type top_dt = memory::data_type::f32;
    fmt = GetOutputFormat(cpu_engine, ip_prim_desc, top_dt, ip_dst_tz);
    std::cout << "[" << out_channels << "]\t" << name.c_str() << "(" << fmt << ")";


  
    // weights
    this->ip_weights_memory = new memory({{{ip_weights_tz},
        memory::data_type::f32, memory::format::oi}, *cpu_engine}, weights);

    // bias
    this->ip_bias_memory = new memory({{{ip_bias_tz},
        memory::data_type::f32, memory::format::x}, *cpu_engine},
        bias); 

    // Prepare dst/top memory
    this->ip_dst_memory = new memory(ip_prim_desc->dst_primitive_desc());

    if ( quantize_type == "int8" ) {
        std::cout << ": record input(to fp32) -";
        std::string nChwxc = get_nChwxc(cpu_engine);
        int fmt_index = GetFormatIndex(nChwxc);
        if ( fmt_index == -1 ) {
            printf("ERROR: Fail to get format of output\n");
            exit(-1);
        }
        reorder = new Reorder(scale_in, batch, in_channel, in_height, in_width);
        memory::format mfmt_nChwxc = formats[fmt_index];
        ip_src_memory = reorder->Init(cpu_engine, bottom, net, mfmt_nChwxc);
    } else {
        ip_src_memory = new memory(bottom);
    }

    // Create Inner Product primitive
    this->ip_fd = new inner_product_forward(*ip_prim_desc, *ip_src_memory,
                *ip_weights_memory, *ip_bias_memory, *ip_dst_memory);
    
    net.push_back(*ip_fd);
    
    if (with_type == "Relu") {
        printf(" [with Relu].\n");
        relu_dst_memory = relu(cpu_engine, ip_prim_desc, ip_dst_memory, 0.0, 0.0, net);

        return relu_dst_memory;
    } else if ( with_type == "Relu6" ) { 
        printf(" Relu6\n");
#if 0
        auto clip_desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_clip,
            ip_prim_desc->dst_primitive_desc().desc(),
            min, max);
        auto clip_prim_desc = eltwise_forward::primitive_desc(clip_desc, *cpu_engine);

        this->clip_dst_memory = new memory(clip_prim_desc.dst_primitive_desc());
        this->clip_fd = new eltwise_forward(clip_prim_desc, 
            *ip_dst_memory,
            *clip_dst_memory);

        net.push_back(*clip_fd);

        return clip_dst_memory;
#endif
    }else {
        printf("\n");
        return ip_dst_memory;
    }
  }

#if 0
private:
  std::string GetOutputFormat(memory::dims &dim) {
    for (int i = 0; i < sizeof(formats) / sizeof(memory::format); ++i) {
      auto md = memory::desc({dim}, memory::data_type::f32, formats[i]);
      auto memory_descriptor = memory::primitive_desc(md, *cpu_engine);
      if (memory::primitive_desc(ip_prim_desc->dst_primitive_desc()) == memory_descriptor) {
          return format_names[i];
      }
    }
    return "unknown";
  }
#endif




public:
  int getInputChannels() { return in_channels; }
  int getOutputChannels() { return out_channels; }
  // Useless: Fc layer is the last layer.
  std::string getQuantizeType() { return "fp32"; }
  float getScaleOut() { return 0.0; }

private:
  std::string name;
  int out_channels;
  int in_channels;
  int in_channel, in_height, in_width;
  float min;
  float max;
  std::string with_type;

  engine *cpu_engine;

  memory *ip_src_memory; 
  memory *ip_weights_memory;
  memory *ip_bias_memory;
  memory *ip_dst_memory;
  inner_product_forward::primitive_desc *ip_prim_desc;
  primitive *ip_fd;

  float scale_in;
  std::string quantize_type;
  Reorder *reorder;

  std::string fmt;

  memory *relu_dst_memory;
  memory *clip_dst_memory;
  primitive *relu_fd;
  primitive *clip_fd;
};

#endif
