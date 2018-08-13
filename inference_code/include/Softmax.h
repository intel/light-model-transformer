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
#ifndef __SOFTMAX_H
#define __SOFTMAX_H
#include <iostream>
#include "mkldnn.hpp"
#include "Format.h"

using namespace mkldnn;

class Softmax {
public:
  Softmax(const std::string &_name, 
               int in_channels,
               const std::string &_quantize_type = "fp32",
               float _scale_in = 1.0):
               name(_name), quantize_type(_quantize_type),
               scale(_scale_in) {
    this->out_channels = in_channels;
    this->in_channels = in_channels;

    softmax_dst_memory = NULL;
    softmax_prim_desc = NULL;
    softmax_fd = NULL;
  }

  ~Softmax() {
    delete softmax_dst_memory;
    delete softmax_prim_desc;
    delete softmax_fd;
  }

  memory *Init(int batch, engine *engine, memory &bottom, std::vector<primitive> &net) {
        this->cpu_engine = engine;

        memory::dims dst_tz = {batch, out_channels};

        int axis = 1;
        auto mem_desc = memory::desc({dst_tz}, memory::data_type::f32, memory::format::nc);
        auto mem_prim_desc = memory::primitive_desc(mem_desc, *cpu_engine);
        auto dst_data = new float[mem_prim_desc.get_size()];
        auto softmax_desc = softmax_forward::desc(prop_kind::forward_scoring, mem_desc, axis);

        this->softmax_prim_desc = new softmax_forward::primitive_desc(softmax_desc, *cpu_engine);
        this->softmax_dst_memory = new memory(mem_prim_desc, dst_data);

        printf("[%d]\t\t%s", this->out_channels, name.c_str());

        if ( quantize_type == "int8" ) {
            std::cout << ": record input(to fp32) -";
            reorder = new Reorder(scale, batch, out_channels, 0, 0);
            fp32_in_memory = reorder->Init(cpu_engine, bottom, net, memory::format::nc);
        } else {
            fp32_in_memory = new memory(bottom);
        }

        this->softmax_fd = new softmax_forward(*softmax_prim_desc, *fp32_in_memory, *softmax_dst_memory);


        net.push_back(*softmax_fd);

        printf("\n");
        return softmax_dst_memory;
    }

public:
  int getInputChannels() { return in_channels; }
  int getOutputChannels() { return out_channels; }

private:
  std::string name;
  int out_channels;
  int in_channels;

  engine *cpu_engine;

  memory *softmax_dst_memory;
  softmax_forward::primitive_desc *softmax_prim_desc;
  primitive *softmax_fd;

  std::string quantize_type;
  float scale;
  memory *fp32_in_memory;
  Reorder *reorder;

};
#endif
