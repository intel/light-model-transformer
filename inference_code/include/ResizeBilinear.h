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

#ifndef __RESIZE_BILINEAR_H
#define __RESIZE_BILINEAR_H
#include <iostream>
#include "mkldnn.hpp"
#include "patch_mkldnn.hpp"
#include "Format.h"

using namespace std;
using namespace mkldnn;

class ResizeBilinear {
public:
  ResizeBilinear(const std::string &_name, 
          int in_channels, 
          int in_h, int in_w, 
          int out_h, int out_w, 
          int h_times, int w_times, 
          const std::string &_align_corners = "False",
          const std::string &_quantize_type = "fp32", // int8 or fp32
          int _scale_out = 0.0):
          name(_name), quantize_type(_quantize_type), scale_out(_scale_out) {
    this->out_channels = in_channels;
    this->in_channels = in_channels;
    this->in_h = in_h;
    this->in_w = in_w;
    if (out_h == 0) this->out_h = in_h * h_times;
    else this->out_h = out_h;
    if (out_w == 0) this->out_w = in_w * w_times;
    else this->out_w = out_w;

    if ( this->out_h == 0 || this->out_w == 0 ) {
        std::cout << std::endl;
        std::cout << "  ERROR[bilinear]:" << std::endl;
        std::cout << "      The out height or width of Layer [" << name;
        std::cout << "] is 0, please check the size of input picture!" << std::endl; 
        std::cout << std::endl;
        exit(-1);
    }

    if (_align_corners == "False") 
        this->align_corners = 0;
    else
        this->align_corners = 1;

    this->fmt = "";

    p_src_memory = NULL;
    p_indices_memory = NULL;
    p_dst_memory = NULL;
    p_prim_desc = NULL;
    bilinear_fd = NULL;
  }

  ~ResizeBilinear() {
    delete p_src_memory;
    delete p_indices_memory;
    delete p_dst_memory;
    delete p_prim_desc;
    delete bilinear_fd;
  }

  // Return the destination/top memory
  memory *Init(int batch, engine *engine, memory &bottom, std::vector<primitive> &net) {
    this->cpu_engine = engine;

    memory::dims bilinear_src_tz = {batch, in_channels, in_h, in_w};
    memory::dims bilinear_dst_tz = {batch, out_channels, out_h, out_w};

    memory::format mfmt_any = memory::format::any;
    memory::data_type bottom_dt = quantize_type == "int8" ? memory::data_type::u8 : memory::data_type::f32;
    memory::data_type top_dt = quantize_type == "int8" ? memory::data_type::u8 : memory::data_type::f32;

    auto bilinear_dst_md = memory::desc({bilinear_dst_tz}, top_dt, mfmt_any);

    // create a resize_bilinear
    auto bilinear_desc = resize_bilinear_forward::desc(prop_kind::forward_scoring,
                bottom.get_primitive_desc().desc(), bilinear_dst_md, align_corners);
    this->p_prim_desc = new resize_bilinear_forward::primitive_desc(bilinear_desc, *cpu_engine);
    this->p_dst_memory = new memory(p_prim_desc->dst_primitive_desc());

    // Seems resize_bilinear does not need to reorder the src
//    this->p_indices_memory = new memory(p_prim_desc->workspace_primitive_desc());
    //this->bilinear_fd = new resize_bilinear_forward(*p_prim_desc, bottom, *p_dst_memory, *p_indices_memory);
    this->bilinear_fd = new resize_bilinear_forward(*p_prim_desc, bottom, *p_dst_memory);

    net.push_back(*bilinear_fd);
    fmt = GetOutputFormat(cpu_engine, p_prim_desc, top_dt, bilinear_dst_tz);
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
  std::string getFormat() { return fmt; }
  std::string getQuantizeType() { return quantize_type; }
  float getScaleOut() { return scale_out; }

private:
  std::string name;
  int out_channels;
  int in_channels;
  int in_h;
  int in_w;
  int out_h;
  int out_w;
  int align_corners;

  std::string quantize_type;
  float scale_out;

  engine *cpu_engine;
  std::string fmt;

  memory *p_src_memory; // used for reorder
  memory *p_indices_memory;
  memory *p_dst_memory;
  resize_bilinear_forward::primitive_desc *p_prim_desc;
  primitive *bilinear_fd;
};

#endif
