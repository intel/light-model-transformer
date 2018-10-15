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
            name(_name), with_type(_with_type) {
        this->out_channels = c;
        this->out_h = h;
        this->out_w = w;
        this->scales = {_scale_in1, _scale_in2};
        this->quantize_types = {_quantize_type1, _quantize_type2};
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
    memory *Init(int batch, engine *engine, std::string bm_fmts1, std::string bm_fmts2, memory bottom1, memory bottom2, std::vector<primitive> &net) {
        this->cpu_engine = engine;
        memory::dims _dst_tz = {batch, out_channels, out_h, out_w};
        std::vector<std::string> bm_fmts;
        std::vector<memory::primitive_desc> srcs_pd;
        std::vector<memory> bottoms = {bottom1, bottom2};
        std::vector<primitive::at> inputs;
        std::vector<float> scale = {1.0, 1.0};
        memory::format fmt[2];
        bool find_fmt;
        bool have_quantize = false;

        bm_fmts.push_back(bm_fmts1);
        bm_fmts.push_back(bm_fmts2);
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

        memory::data_type bottom_dt = memory::data_type::f32;
#if 0
        for ( int i = 0; i < 2; i++ ) {
            if (quantize_type[i] == "int8") {
                bottom_dt = memory::data_type::u8;
                have_quantize = true;
            }
        }
#endif
        for ( int i = 0; i < 2; i++ ) {
            auto desc = memory::desc(_dst_tz, bottom_dt, fmt[i]);
            auto mpd = memory::primitive_desc(desc, *cpu_engine);
            srcs_pd.push_back(mpd);
        }


        int f = fmt[0]<fmt[1] ? 0:1;
        auto _desc = memory::desc({_dst_tz}, bottom_dt, fmt[f]);
        this->p_prim_desc = new sum::primitive_desc(_desc, scale, srcs_pd);
        p_dst_memory = new memory(p_prim_desc->dst_primitive_desc());

#if 1
        int quantize_num = -1;
        for ( int i = 0; i < scales.size(); i ++ ) {
            if ( quantize_types[i] == "int8" ) {
                quantize_num ++;
                reorders.push_back(new Reorder(scales[i], batch, out_channels, out_h, out_w));
                fp32_in_memorys.push_back(reorders[quantize_num]->Init(cpu_engine, bottoms[i], net, memory::format::nhwc));
                inputs.push_back( *fp32_in_memorys[quantize_num] );
            } else {
                inputs.push_back( bottoms[i] );
            }
        }
#else
        // Unimplemented
        for ( int i = 0; i < 2; i++ ) {
//            scale[i] = scales[i];
        }
        if ( quantize_type1 == "int8" ) {
            printf("input1 -> int8\n");
            Reorder reorder(scales[0]);
            fp32_in_memory1 = reorder.Init(bottom1, *fp32_in_memory1, net);
        }
        if ( quantize_type2 == "int8" ) {
            printf("input2 -> int8\n");
            //Reorder reorder(scales[1]);
            Reorder reorder(100);
            fp32_in_memory2 = reorder.Init(bottom2, *fp32_in_memory2, net);
        }
#endif
        _fd = new sum(*p_prim_desc, inputs, *p_dst_memory);

        net.push_back(*_fd);

        sum_fmt = GetOutputFormat(cpu_engine, p_prim_desc, bottom_dt, _dst_tz);
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

    std::vector<std::string> quantize_types;
    std::vector<float> scales;
    std::vector<Reorder*> reorders;
    std::vector<memory*> fp32_in_memorys;

    engine *cpu_engine;
    memory *p_dst_memory;
    memory *relu_dst_memory;
    sum::primitive_desc *p_prim_desc;
    primitive *_fd;
    primitive *relu_fd;
    std::vector<std::vector<int> > in_dims;
};

#endif
