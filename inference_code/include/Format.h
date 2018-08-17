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

#ifndef __FORMAT_H
#define __FORMAT_H
#include "mkldnn.hpp"
#include "patch_mkldnn.hpp"

using namespace mkldnn;

#if 1
memory::format formats_2d[] = {
    memory::format::format_undef,
    memory::format::oi,
    memory::format::io
};

const char *format_2d_names[] = {
    "format_undef",
    "oi",
    "io"
};
#endif
memory::format formats[] = {
    memory::format::format_undef,
    memory::format::nchw,
    memory::format::nhwc,
    memory::format::chwn,
    memory::format::nChw8c,
    memory::format::nChw16c,
    memory::format::oi,
    memory::format::io,
    memory::format::oihw,
    memory::format::ihwo,
    memory::format::hwio,
    memory::format::oIhw8i,
    memory::format::oIhw16i,
    memory::format::OIhw8i8o,
    memory::format::OIhw16i16o,
    memory::format::OIhw8o8i,
    memory::format::OIhw16o16i,
    memory::format::IOhw16o16i,
    memory::format::OIhw8i16o2i,
    memory::format::OIhw8o16i2o,
    memory::format::OIhw4i16o4i,
    memory::format::Oihw8o,
    memory::format::Oihw16o,
    memory::format::Ohwi8o,
    memory::format::Ohwi16o,
    memory::format::OhIw16o4i,
    memory::format::goihw,
    memory::format::hwigo,
    memory::format::gOIhw8i8o,
    memory::format::gOIhw16i16o,
    memory::format::gOIhw8i16o2i,
    memory::format::gOIhw8o16i2o,
    memory::format::gOIhw4i16o4i,
    memory::format::gOihw8o,
    memory::format::gOihw16o,
    memory::format::gOhwi8o,
    memory::format::gOhwi16o,
    memory::format::gOIhw8o8i,
    memory::format::gOIhw16o16i,
    memory::format::gIOhw16o16i,
    memory::format::gOhIw16o4i,
};
const char *format_names[] = {
    "format_undef",
    "nchw",
    "nhwc",
    "chwn",
    "nChw8c",
    "nChw16c",
    "oi",
    "io",
    "oihw",
    "ihwo",
    "hwio",
    "oIhw8i",
    "oIhw16i",
    "OIhw8i8o",
    "OIhw16i16o",
    "OIhw8o8i",
    "OIhw16o16i",
    "IOhw16o16i",
    "OIhw8i16o2i",
    "OIhw8o16i2o",
    "OIhw4i16o4i",
    "Oihw8o",
    "Oihw16o",
    "Ohwi8o",
    "Ohwi16o",
    "OhIw16o4i",
    "goihw",
    "hwigo",
    "gOIhw8i8o",
    "gOIhw16i16o",
    "gOIhw8i16o2i",
    "gOIhw8o16i2o",
    "gOIhw4i16o4i",
    "gOihw8o",
    "gOihw16o",
    "gOhwi8o",
    "gOhwi16o",
    "gOIhw8o8i",
    "gOIhw16o16i",
    "gIOhw16o16i",
    "gOhIw16o4i"
};

int GetFormatIndex(std::string fmt_name) {
    for (int i = 0; i < sizeof(formats) / sizeof(memory::format); ++i) {
        if ( !strcmp(fmt_name.c_str(), format_names[i]) ) return i;
    }
    return -1;
}

std::string GetOutputFormat(engine *engine, convolution_forward::primitive_desc *p_prim_desc, 
                            memory::data_type type, memory::dims &dim) {
    for (int i = 0; i < sizeof(formats) / sizeof(memory::format); ++i) {
        auto md = memory::desc({dim}, type, formats[i]);
        auto memory_descriptor = memory::primitive_desc(md, *engine);
        if (memory::primitive_desc(p_prim_desc->dst_primitive_desc()) == memory_descriptor)
            return format_names[i];
    }
    return "unknown";
}

std::string GetOutputFormat(engine *engine, pooling_forward::primitive_desc *p_prim_desc, 
                            memory::data_type type, memory::dims &dim) {
    for (int i = 0; i < sizeof(formats) / sizeof(memory::format); ++i) {
        auto md = memory::desc({dim}, type, formats[i]);
        auto memory_descriptor = memory::primitive_desc(md, *engine);
        if (memory::primitive_desc(p_prim_desc->dst_primitive_desc()) == memory_descriptor)
            return format_names[i];
    }
    return "unknown";
}

std::string GetOutputFormat(engine *engine, extract_image_patches_forward::primitive_desc *p_prim_desc, 
                            memory::data_type type, memory::dims &dim) {
    for (int i = 0; i < sizeof(formats) / sizeof(memory::format); ++i) {
        auto md = memory::desc({dim}, type, formats[i]);
        auto memory_descriptor = memory::primitive_desc(md, *engine);
        if (memory::primitive_desc(p_prim_desc->dst_primitive_desc()) == memory_descriptor)
            return format_names[i];
    }
    return "unknown";
}

std::string GetOutputFormat(engine *engine, resize_bilinear_forward::primitive_desc *p_prim_desc, 
                            memory::data_type type, memory::dims &dim) {
    for (int i = 0; i < sizeof(formats) / sizeof(memory::format); ++i) {
        auto md = memory::desc({dim}, type, formats[i]);
        auto memory_descriptor = memory::primitive_desc(md, *engine);
        if (memory::primitive_desc(p_prim_desc->dst_primitive_desc()) == memory_descriptor)
            return format_names[i];
    }
    return "unknown";
}

std::string GetOutputFormat(engine *engine, deconvolution_forward::primitive_desc *p_prim_desc,
                             memory::data_type type, memory::dims &dim) {
     for (int i = 0; i < sizeof(formats) / sizeof(memory::format); ++i) {
         auto md = memory::desc({dim}, type, formats[i]);
         auto memory_descriptor = memory::primitive_desc(md, *engine);
         if (memory::primitive_desc(p_prim_desc->dst_primitive_desc()) == memory_descriptor)
             return format_names[i];
     }
     return "unknown";
 }



std::string GetOutputFormat(engine *engine, inner_product_forward::primitive_desc *p_prim_desc, 
                            memory::data_type type, memory::dims &dim) {
    for (int i = 0; i < sizeof(formats) / sizeof(memory::format); ++i) {
        auto md = memory::desc({dim}, type, formats_2d[i]);
        auto memory_descriptor = memory::primitive_desc(md, *engine);
        if (memory::primitive_desc(p_prim_desc->dst_primitive_desc()) == memory_descriptor)
            return format_2d_names[i];
    }
    return "unknown";
}

std::string get_nChwxc(engine *engine) { 
    auto src_md = memory::desc({1, 3, 1, 1}, memory::data_type::f32, memory::format::any);
    auto dst_md = memory::desc({1, 32, 1, 1}, memory::data_type::f32, memory::format::any);
    auto weights_md = memory::desc({32, 3, 1, 1}, memory::data_type::f32, memory::format::any);
    auto bias_md = memory::desc({32}, memory::data_type::f32, memory::format::any);

    // Fake convlution, only needed by GetOutputFormat.
    convolution_forward::primitive_desc *p_prim_desc;
    auto desc = convolution_forward::desc(prop_kind::forward, convolution_direct, 
        src_md, weights_md, bias_md, dst_md, 
        {1, 1}, {0, 0}, {0, 0}, padding_kind::zero);
    p_prim_desc = new convolution_forward::primitive_desc(desc, *engine);

    memory::dims dst_tz = {1, 32, 1, 1};
    std::string nChwxc = GetOutputFormat(engine, p_prim_desc, memory::data_type::f32, dst_tz);
    
    return nChwxc;
}



#endif
