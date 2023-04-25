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
            dnnl::memory::desc dst_md{src_md.get_dims(), dnnl::memory::data_type::f32, dnnl::memory::dims{}};
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

        if (tensor.get_desc().get_data_type() != dnnl::memory::data_type::f32)
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

    static constexpr int num_parameters = 2;
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
    MinMax output;

    static QuantizationFactors FromVector(const std::vector<float>& factors)
    {
        if (factors.size() != QuantizationFactors::num_parameters)
        {
            throw std::runtime_error("Invalid length of quantization factors vector.");
        }
        
        return QuantizationFactors {
            {factors.at(0), factors.at(1)},
            {factors.at(2), factors.at(3)},
            {factors.at(4), factors.at(5)},
            {factors.at(6), factors.at(7)}
        };
    }

    std::vector<float> AsVector() const
    {
        return std::vector<float>
        {
            qkv.min, qkv.max,
            attention_out.min, attention_out.max,
            intermediate.min, intermediate.max,
            output.min, output.max
        };
    }

    static constexpr int num_parameters = MinMax::num_parameters * 4;
};

std::ostream& operator<< (std::ostream& os, const QuantizationFactors& v)
{
    os << v.qkv << ' ' << v.attention_out << ' ' << v.intermediate << ' ' << v.output;
    return os;
}

std::istream& operator>> (std::istream& is, QuantizationFactors& v)
{
    is >> v.qkv >> v.attention_out >> v.intermediate >> v.output;
    return is;
}


float computeQuantizationScale(dnnl::memory::data_type data_type, float min, float max)
{
    using dt = dnnl::memory::data_type;
    switch(data_type)
    {
        case dt::s8:
            return std::max(std::abs(min), std::abs(max)) / static_cast<float>(std::numeric_limits<int8_t>::max());
        case dt::u8:
            return                    std::abs(max - min) / static_cast<float>(std::numeric_limits<uint8_t>::max());
        case dt::s32:
            return std::max(std::abs(min), std::abs(max)) / static_cast<float>(std::numeric_limits<int32_t>::max());
        default:
            return dnnl_wrappers::BuildAttrs::noScale;
    }
}

float computeQuantizationScale(dnnl::memory::data_type data_type, const float* p, size_t size)
{
    auto min_max = std::minmax_element(p, p+size);
    return computeQuantizationScale(data_type, *min_max.first, *min_max.second);
}

float computeQuantizationScale(dnnl::memory::data_type data_type, const dnnl::memory& mem, dnnl::stream wait_stream = dnnl::stream{}) {
    assert(mem.get_desc().get_data_type() == dnnl::memory::data_type::f32);
    if (wait_stream) {
        wait_stream.wait();
    }
    MemoryAccessor<float> mem_acc(mem);
    return computeQuantizationScale(data_type, mem_acc.Data(), mem.get_desc().get_size() / sizeof(float));
}

#endif
