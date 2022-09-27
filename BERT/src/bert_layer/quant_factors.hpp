// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LIBRARIES_AI_PERFORMANCE_MODELS_BERT_QUANT_FACTORS_HPP_
#define LIBRARIES_AI_PERFORMANCE_MODELS_BERT_QUANT_FACTORS_HPP_

#include "dnnl.hpp"

#include <limits>
#include <iostream>

struct MinMax {
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::lowest();

    void Update(const dnnl::memory& tensor)
    {
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
    os << v.min << ' ' << v.max;
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
