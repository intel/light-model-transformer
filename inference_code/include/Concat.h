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
#ifndef __CONCAT_H
#define __CONCAT_H
#include <iostream>
#include <assert.h>
#include "mkldnn.hpp"
#include "Format.h"
#include "Reorder.h"

using namespace mkldnn;


class Concat{
public:
    Concat(const std::string &_name,
            std::vector<std::vector<int> > in_dims,
            char concat_dim_char,
            std::vector<std::string> _quantize_types,
            std::vector<float> _scales):
            name(_name) { 
        this->concat_dim = 0;
        this->concat_dim_char = concat_dim_char;
        this->out_channels = 0;
        this->out_h = 0;
        this->out_w = 0;
        this->in_dims = in_dims;

        this->end_index = in_dims.size();

        this->quantize_type1 = _quantize_types[0];
        this->scale_in1 = _scales[0];
        this->quantize_type2 = _quantize_types[1];
        this->scale_in2 = _scales[1];
        if ( this->end_index > 2 ) {
            this->quantize_type3 = _quantize_types[2];
            this->scale_in3 = _scales[2];
            if ( this->end_index > 3 ) {
                this->quantize_type4 = _quantize_types[3];
                this->scale_in4 = _scales[3];
            }
        }

        out_channels = in_dims[0][0];
        out_h = in_dims[0][1];
        out_w = in_dims[0][2];
        for ( int i = 1; i < end_index; i++ ) {
            if ( concat_dim_char == 'c' ) {
                if ( in_dims[0][1] != in_dims[i][1] ) {
                    printf("\n    ERROR: Different height of concat' input [%d/%d]\n", in_dims[0][1], in_dims[i][1]); exit(-1);
                }
                if ( in_dims[0][2] != in_dims[i][2] ) {
                    printf("\n    ERROR: Different width of concat' input [%d/%d]\n", in_dims[0][2], in_dims[i][2]); exit(-1);
                }
                concat_dim = 1;
                out_channels += in_dims[i][0];
            } else if ( concat_dim_char == 'h' ) {
                if ( in_dims[0][0] != in_dims[i][0] ) {
                    printf("\n    ERROR: Different channel of concat' input [%d/%d]\n", in_dims[0][1], in_dims[i][1]); exit(-1);
                }
                if ( in_dims[0][2] != in_dims[i][2] ) {
                    printf("\n    ERROR: Different width of concat' input [%d/%d]\n", in_dims[0][2], in_dims[i][2]); exit(-1);
                }
                concat_dim = 2;
                out_h += in_dims[i][1];
            } else if ( concat_dim_char == 'w' ) {
                if ( in_dims[0][0] != in_dims[i][0] ) {
                    printf("\n    ERROR: Different channel of concat' input [%d/%d]\n", in_dims[0][1], in_dims[i][1]); exit(-1);
                }
                if ( in_dims[0][1] != in_dims[i][1] ) {
                    printf("\n    ERROR: Different height of concat' input [%d/%d]\n", in_dims[0][1], in_dims[i][1]); exit(-1);
                }
                concat_dim = 3;
                out_w += in_dims[i][2];
            } else {
                printf("\n  ERROE: Wrong concat dimension type.\n");
                exit(-1);
            }
        }

        if ( this->out_h == 0 || this->out_w == 0 ) {
            std::cout << std::endl;
            std::cout << "  ERROR[Concat]:" << std::endl;
            std::cout << "      The out height or width of Layer [" << name;
            std::cout << "] is 0, please check the size of input picture!" << std::endl; 
            std::cout << std::endl;
            exit(-1);
        }

        p_dst_memory = NULL;
        concat_fd = NULL;

    }

    ~Concat() {
        delete p_dst_memory;
        delete concat_fd;
    }
    memory *Init(int batch, engine *engine, std::vector<std::string> bm_fmts, std::vector<memory>& bottoms, std::vector<primitive> &net) {

        this->cpu_engine = engine;
        memory::dims concat_dst_tz = {batch, out_channels, out_h, out_w};
        std::vector<memory::primitive_desc> srcs_pd;
        std::vector<memory> srcs;
        memory::format fmt[end_index];
        bool find_fmt;

        memory::format mfmt_any = memory::format::any;
        memory::format mfmt_nhwc = memory::format::nhwc;

        memory::data_type bottom_dt = memory::data_type::f32;
        memory::data_type top_dt = memory::data_type::f32;

        std::vector<primitive::at> inputs;

        // Prepare input of quantize(int8) or fp32 
        if ( quantize_type1 == "int8" ) {
            reorder1 = new Reorder(scale_in1, batch, in_dims[0][0], in_dims[0][1], in_dims[0][2]);
            fp32_in_memory1 = reorder1->Init(cpu_engine, bottoms[0], net, mfmt_nhwc);
            inputs.push_back( *fp32_in_memory1 );
        } else inputs.push_back( bottoms[0] );

        if ( quantize_type2 == "int8" ) {
            reorder2 = new Reorder(scale_in2, batch, in_dims[1][0], in_dims[1][1], in_dims[1][2]);
            fp32_in_memory2 = reorder2->Init(cpu_engine, bottoms[1], net, mfmt_nhwc);
            inputs.push_back( *fp32_in_memory2 );
        } else inputs.push_back( bottoms[1] );

        if ( end_index >= 2 ) {
            if ( quantize_type3 == "int8" ) {
                reorder3 = new Reorder(scale_in3, batch, in_dims[2][0], in_dims[2][1], in_dims[2][2]);
                fp32_in_memory3 = reorder3->Init(cpu_engine, bottoms[2], net, mfmt_nhwc);
                inputs.push_back( *fp32_in_memory3 );
            } else inputs.push_back( bottoms[2] );
        }

        if ( end_index >= 3 ) {
            if ( quantize_type4 == "int8" ) {
                reorder4 = new Reorder(scale_in4, batch, in_dims[3][0], in_dims[3][1], in_dims[3][2]);
                fp32_in_memory4 = reorder4->Init(cpu_engine, bottoms[3], net, mfmt_nhwc);
                inputs.push_back( *fp32_in_memory4 );
            } else inputs.push_back( bottoms[3] );
        }

        // Prepare format of input
        for (int i = 0, fmt_index = 0; i < end_index; i++) {
            find_fmt = false;
            for (int j = 0; j < sizeof(formats) / sizeof(memory::format); ++j) {
                if ( bm_fmts[i] == format_names[j] ) {
                    fmt[fmt_index++] = formats[j];
                    r_fmt = bm_fmts[i];
                    find_fmt = true;
                    continue;
                }
            }
            if ( !find_fmt ) {
                std::cout << "  ERROR: Unknown format from input [" << i << "] into concat." << std::endl;
                exit(-1);
            }
        }

        for ( int i = 0, j = 0; i < end_index; i++, j++ ) {
            auto desc = memory::desc({batch, in_dims[i][0], in_dims[i][1], in_dims[i][2]}, \
                                        bottom_dt, fmt[j]);
            auto mpd = memory::primitive_desc(desc, *cpu_engine);
            srcs_pd.push_back(mpd);
        }

        auto concat_desc = memory::desc({concat_dst_tz}, top_dt, fmt[end_index-1]);
        // dim 一定要和合并的维度序号一致
        this->p_prim_desc = new concat::primitive_desc(concat_desc, concat_dim, srcs_pd); // p_prim_desc
        if (!p_dst_memory)
            p_dst_memory = new memory(p_prim_desc->dst_primitive_desc());

        concat_fd = new concat(*p_prim_desc, inputs, *p_dst_memory);
        net.push_back(*concat_fd);

        std::cout << "[" << out_h << "*" << out_w << "*" << out_channels << "]\t" << name.c_str() << "(" << r_fmt << ")\n";
        return p_dst_memory;
    }


public:
    int getOutputChannels() { return out_channels; }
    int getOutputHeight() { return out_h; }
    int getOutputWidth() { return out_w; }
    std::string getQuantizeType() { return "fp32"; }
    std::string getFormat() { return r_fmt; }

private:
    std::string name;
    int out_channels;
    int out_h;
    int out_w;
    int concat_dim;
    int concat_dim_char;
    int begin_index;
    int end_index;

    std::string quantize_type1, quantize_type2, quantize_type3, quantize_type4;
    float scale_in1, scale_in2, scale_in3, scale_in4;
    memory *fp32_in_memory1, *fp32_in_memory2, *fp32_in_memory3, *fp32_in_memory4;
    Reorder *reorder1, *reorder2, *reorder3, *reorder4;

    std::string r_fmt;
    engine *cpu_engine;
    memory *p_dst_memory;
    concat::primitive_desc *p_prim_desc;
    primitive *concat_fd;
    std::vector<std::vector<int> > in_dims;
};

#endif
