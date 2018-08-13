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
#ifndef __REORDER_H
#define __REORDER_H
#include <iostream>
#include "mkldnn.hpp"
#include "Format.h"

using namespace mkldnn;

class Reorder {
public:
    //memory::dims conv_weights_tz = {out_dim2, in_dim2, kernel_h, kernel_w};
    // dim2: batch of convolution or out_channels of weights
    // dim2: out_channels of convolution or in_channels of weights
    // dim3: height of convolution or kernel_h of weights 
    // dim3: width of convolution or kernel_w of weights
    Reorder(float _scale = 0.0, int _dim1 = 0, int _dim2 = 0, int _dim3 = 0, int _dim4 = 0)
    {
        this->dim1 = _dim1;
        this->dim2 = _dim2;
        this->dim3 = _dim3;
        this->dim4 = _dim4;
        this->scale = _scale;
        if ( _dim3 == 0 ) {
            this->user_dst = new float[dim1 * dim2];
            this->user_dst_tz = {dim1, dim2};
        } else {
            this->user_dst = new float[dim1 * dim2 * dim3 * dim4];
            this->user_dst_tz = {dim1, dim2, dim3, dim4};
        }

        p_dst_memory = NULL;
    }

    ~Reorder() {
        delete[] user_dst;
        delete p_dst_memory;
    }

    // fp32 -> int8, execute right now
    memory *Init(memory user_src_memory, memory &user_dst_memory, bool need_quantize=false) {
        std::vector<primitive> tmp_net;

        if ( need_quantize ) {
            primitive_attr dst_attr;
            dst_attr.set_int_output_round_mode(round_mode::round_nearest);
            std::vector<float> dst_scales = { scale };
            dst_attr.set_output_scales(0, dst_scales);
            auto dst_reorder_pd = reorder::primitive_desc(user_src_memory.get_primitive_desc(),
                    user_dst_memory.get_primitive_desc(), dst_attr);
            tmp_net.push_back(reorder(dst_reorder_pd, user_src_memory, user_dst_memory));
        } else {
            tmp_net.push_back(reorder(user_src_memory, user_dst_memory));
        }

        stream(stream::kind::eager).submit(tmp_net).wait();
    
        return &user_dst_memory;
    }

    // fp32 -> int8, don't execute
    memory *Init(memory &user_src_memory, memory &user_dst_memory, std::vector<primitive> &net) {
        primitive_attr dst_attr;
        dst_attr.set_int_output_round_mode(round_mode::round_nearest);
        std::vector<float> dst_scales = { scale };
        dst_attr.set_output_scales(0, dst_scales);
        auto dst_reorder_pd = reorder::primitive_desc(user_src_memory.get_primitive_desc(),
                    user_dst_memory.get_primitive_desc(), dst_attr);
        net.push_back(reorder(dst_reorder_pd, user_src_memory, user_dst_memory));
        return &user_dst_memory;
    }

    // int8 -> fp32
    memory *Init(engine *engine, memory &bottom, std::vector<primitive> &net, memory::format mfmt) {
        p_dst_memory = new memory({{{user_dst_tz},
                memory::data_type::f32, mfmt}, *engine }, user_dst);

        primitive_attr dst_attr;
        dst_attr.set_int_output_round_mode(round_mode::round_nearest);
        std::vector<float> dst_scales = { 1 / scale };
        dst_attr.set_output_scales(0, dst_scales);
        auto dst_reorder_pd = reorder::primitive_desc(bottom.get_primitive_desc(),
                p_dst_memory->get_primitive_desc(), dst_attr);

        net.push_back(reorder(dst_reorder_pd, bottom, *p_dst_memory));
    
        return p_dst_memory;
    }

    // int8 -> fp32
    memory *Init(engine *engine, memory &bottom, std::vector<primitive> &net) {
//        std::cout << name.c_str() << "(int8 -> fp32)" << std::endl;
        p_dst_memory = new memory({{{user_dst_tz},
                memory::data_type::f32, memory::format::oihw}, *engine }, user_dst);

        primitive_attr dst_attr;
        dst_attr.set_int_output_round_mode(round_mode::round_nearest);
        std::vector<float> dst_scales = { 1 / scale };
        dst_attr.set_output_scales(0, dst_scales);
        auto dst_reorder_pd = reorder::primitive_desc(bottom.get_primitive_desc(),
                p_dst_memory->get_primitive_desc(), dst_attr);

        net.push_back(reorder(dst_reorder_pd, bottom, *p_dst_memory));
    
        return p_dst_memory;
    }


private:
    std::string name;
    int dim1, dim2, dim3, dim4;
    float scale;
    float *user_dst;
    engine *cpu_engine;
    memory::dims user_dst_tz;

    memory *p_dst_memory;
};

#endif
