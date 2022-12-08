// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LIBRARIES_AI_PERFORMANCE_MODELS_BERT_QUANT_FACTORS_HPP_
#define LIBRARIES_AI_PERFORMANCE_MODELS_BERT_QUANT_FACTORS_HPP_

#include "dnnl.hpp"

#include <limits>
#include <iomanip>
#include <iostream>

struct MinMax {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();

    void Update(const dnnl::memory& mem)
    {
        // reorder to plain mem
        auto tensor = [](dnnl::memory src_mem) -> dnnl::memory {
            auto src_md = src_mem.get_desc();
            dnnl::memory::desc dst_md{src_md.dims(), dnnl::memory::data_type::f32, dnnl::memory::dims{}};
            if (dst_md == src_md) {
                return src_mem;
            }

            auto eng = src_mem.get_engine();
            dnnl::stream stm{eng};
            dnnl::memory dst_mem{dst_md, eng};
            dnnl::reorder{src_mem, dst_mem}.execute(stm, src_mem, dst_mem);
            stm.wait();
            return dst_mem;
        }(mem);

        if (tensor.get_desc().data_type() != dnnl::memory::data_type::f32)
        {
            throw std::invalid_argument("MinMax requires tensors for quantization factor calibration to have "
                                        "FP32 data type.");
        }
        float* p = (float*)tensor.get_data_handle();
        auto size = tensor.get_desc().get_size() / sizeof(float);
        
        auto min_element = *std::min_element(p, p+size);
        auto max_element = *std::max_element(p, p+size);

        min = std::min(min, min_element);
        max = std::max(max, max_element);
    }
};

std::ostream& operator<< (std::ostream& os, const MinMax& v)
{
    const auto prec = std::numeric_limits<float>::max_digits10;
    const auto old_prec = os.precision(prec);
    os << std::setw(prec + 2) << v.min << std::setw(prec + 2) << v.max;
    os.precision(old_prec);
    return os;
}

std::istream& operator>> (std::istream& is, MinMax& v)
{
    is >> v.min >> v.max;
    return is;
}


struct QuantizationFactors
{
    MinMax qkv;
    MinMax attention_out;
    MinMax intermediate;
    MinMax intermediate_post;

    static float From(dnnl::memory& mem, dnnl::stream& stm);
};

std::ostream& operator<< (std::ostream& os, const QuantizationFactors& v)
{
    os << v.qkv << ' ' << v.attention_out << ' ' << v.intermediate << ' ' << v.intermediate_post;
    return os;
}

std::istream& operator>> (std::istream& is, QuantizationFactors& v)
{
    is >> v.qkv >> v.attention_out >> v.intermediate >> v.intermediate_post;
    return is;
}

#endif
