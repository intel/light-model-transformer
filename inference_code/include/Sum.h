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
#ifndef __SUM_H
#define __SUM_H
#include <iostream>
#include <assert.h>
#include "mkldnn.hpp"
#include "Format.h"
#include "Reorder.h"

using namespace mkldnn;


class Sum{
public:
    Sum(const std::string &_name,
            int c, int h, int w,
            const std::string &_with_type = "none",
            const std::string &_quantize_type1 = "fp32", // int8 or fp32
            const std::string &_quantize_type2 = "fp32", // int8 or fp32
            float _scale_in1 = 1.0, float _scale_in2 = 1.0):
            name(_name), with_type(_with_type), 
            quantize_type1(_quantize_type1), quantize_type2(_quantize_type2) {
        this->out_channels = c;
        this->out_h = h;
        this->out_w = w;
        this->scales = {_scale_in1, _scale_in2};
        //printf("%s - oh:%d, ow:%d, oc:%d\n", name.c_str(), out_h, out_w, out_channels);

        if ( this->out_h == 0 || this->out_w == 0 ) {
            std::cout << std::endl;
            std::cout << "  ERROR[Sum]:" << std::endl;
            std::cout << "      The out height or width of Layer [" << name;
            std::cout << "] is 0, please check the size of input picture!" << std::endl; 
            std::cout << std::endl;
            exit(-1);
        }

        p_dst_memory = NULL;
        _fd = NULL;

        this->sum_fmt = "";
    }

    ~Sum() {
        delete p_dst_memory;
        delete _fd;
    }
    memory *Init(int batch, engine *engine, std::string bm_fmts1, std::string bm_fmts2, memory bottoms1, memory bottoms2, std::vector<primitive> &net) {
        this->cpu_engine = engine;
        memory::dims _dst_tz = {batch, out_channels, out_h, out_w};
        std::vector<std::string> bm_fmts;
        std::vector<memory::primitive_desc> srcs_pd;
        std::vector<memory> srcs;
        memory::format fmt[2];
        bool find_fmt;

        bm_fmts.push_back(bm_fmts1);
        bm_fmts.push_back(bm_fmts2);
        sum_fmt = bm_fmts2;
        for (int i = 0, fmt_index = 0; i < 2; i++) {
            find_fmt = false;
            for (int j = 0; j < sizeof(formats) / sizeof(memory::format); ++j) {
                if ( bm_fmts[i] == format_names[j] ) {
                    fmt[fmt_index++] = formats[j];
                    find_fmt = true;
                    continue;
                }
            }
            if ( !find_fmt ) {
                std::cout << "  ERROR: Unknown format from input [" << i << "] into concat." << std::endl;
                exit(-1);
            }
        }

#if 0
        memory::data_type bottom_dt = quantize_type == "int8" ? memory::data_type::u8 : memory::data_type::f32;
        memory::data_type top_dt = quantize_type == "int8" ? memory::data_type::u8 : memory::data_type::f32;
#else
        memory::data_type bottom_dt = memory::data_type::f32;
        memory::data_type top_dt = memory::data_type::f32;
#endif

        for ( int i = 0, j = 0; i < 2; i++, j++ ) {
            auto desc = memory::desc(_dst_tz, bottom_dt, fmt[j]);
            auto mpd = memory::primitive_desc(desc, *cpu_engine);
            srcs_pd.push_back(mpd);
        }

        auto _desc = memory::desc({_dst_tz}, top_dt, fmt[1]);
        std::vector<float> scale = {1.0, 1.0};
        this->p_prim_desc = new sum::primitive_desc(_desc, scale, srcs_pd);
        p_dst_memory = new memory(p_prim_desc->dst_primitive_desc());

        memory::format mfmt_nhwc = memory::format::nhwc;
        std::vector<primitive::at> inputs;

        if ( quantize_type1 == "int8" ) {
            reorder1 = new Reorder(scales[0], batch, out_channels, out_h, out_w);
            fp32_in_memory1 = reorder1->Init(cpu_engine, bottoms1, net, mfmt_nhwc);
            inputs.push_back( *fp32_in_memory1 );
        } else {
            inputs.push_back( bottoms1 );
        }

        if ( quantize_type2 == "int8" ) {
            reorder2 = new Reorder(scales[1], batch, out_channels, out_h, out_w);
            fp32_in_memory2 = reorder2->Init(cpu_engine, bottoms2, net, mfmt_nhwc);
            inputs.push_back( *fp32_in_memory2 );
        } else {
            inputs.push_back( bottoms2 );
        }

        _fd = new sum(*p_prim_desc, inputs, *p_dst_memory);
        net.push_back(*_fd);
        std::cout << "[" << out_h << "*" << out_w << "*" << out_channels << "]\t" << name.c_str() << "(" << sum_fmt << ")";

        if ( with_type == "Relu" ) { 
            printf(" [with Relu]\n");
            auto relu_desc = eltwise_forward::desc(prop_kind::forward,
                algorithm::eltwise_relu,
                p_prim_desc->dst_primitive_desc().desc(), 0.0);
            auto relu_prim_desc = eltwise_forward::primitive_desc(relu_desc, *cpu_engine);

            this->relu_dst_memory = new memory(relu_prim_desc.dst_primitive_desc());
            relu_fd = new eltwise_forward(relu_prim_desc, 
                *p_dst_memory,
                *relu_dst_memory);

            net.push_back(*relu_fd);

            return relu_dst_memory;
        } else {
            printf("\n");
            return p_dst_memory;
        }
    }


public:
    int getOutputChannels() { return out_channels; }
    int getOutputHeight() { return out_h; }
    int getOutputWidth() { return out_w; }
    std::string getFormat() { return sum_fmt; }
    std::string getQuantizeType() { return "fp32"; }

private:
    std::string name;
    std::string sum_fmt;
    std::string with_type;
    int out_channels;
    int out_h;
    int out_w;

    std::string quantize_type1, quantize_type2;
    std::vector<float> scales;
    memory *fp32_in_memory1, *fp32_in_memory2;
    Reorder *reorder1, *reorder2;

    engine *cpu_engine;
    memory *p_dst_memory;
    memory *relu_dst_memory;
    sum::primitive_desc *p_prim_desc;
    primitive *_fd;
    primitive *relu_fd;
    std::vector<std::vector<int> > in_dims;
};

#endif
