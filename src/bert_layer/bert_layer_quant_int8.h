// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_LAYER_H_
#define BERT_LAYER_H_

#include "my_types.h"
#include "dnnl_common.h"
#include "bert_context.h"
#include "dnnl_attr.hpp"
#include "dnnl_data.hpp"
#include "dnnl_ops.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cassert>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>


#define QUANT_INT8
//#define dynamic_quant

#define BFLOAT16_ATTENTION 0

namespace dnnl_wrappers {
// TODO(rfsaliev) Remove the temporary helper to attach dnnl::memory to hpj::Matrix<> data
template <class T>
dnnl::memory AttachMemory(const dnnl::engine& eng, hpj::Matrix<T>& data, bool trans = false) {
    dnnl::memory::dims dims{data.Rows(), data.Cols()};
    return dnnl_wrappers::AttachMemory(eng, dims, data.Data(), trans);
}

/// Attach dnnl::memory to memory buffer using specified layout
template <class T>
dnnl::memory AttachMemory(const dnnl::engine& eng, T* data, dnnl::memory::desc layout) {
    // enforce memory data type to be equal to matrix data type
    layout.data.data_type = dnnl::memory::convert_to_c(DnnlDataType<T>::value);
    return dnnl::memory{layout, eng, data};
}

/// Attach dnnl::memory to matrix using specified layout but keeping origin data_type
template <class T>
dnnl::memory AttachMemory(const dnnl::engine& eng, hpj::Matrix<T>& data, dnnl::memory::desc layout) {
    // enforce memory data type to be equal to matrix data type
    layout.data.data_type = dnnl::memory::convert_to_c(DnnlDataType<T>::value);
    assert(data.Rows() * data.Cols() * sizeof(T) == layout.get_size());

    return AttachMemory(eng, data.Data(), layout);
}
} // namespace dnnl_wrappers

class BertLayer
{
#ifdef QUANT_INT8
    using input_t = int8_t;
#else
    using input_t = float;
#endif

public:
    // All BERT models use same head size - 64
    // * Base: hiddenSize = 768, heads = 12
    // * Large: hiddenSize = 1024, heads = 16
    static constexpr int head_size = 64;

    // hiddenSize 768 Hidden layer neurons, number of hidden units
    // intermediateSize 3072 feed-forward/filter size dimension 4*hiddensize 
    BertLayer(BertContext &_ctx, int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072) :
    ctx(_ctx) {
        this->maxTokenSize = maxTokenSize;
        this->hiddenSize = hiddenSize;
        this->intermediateSize = intermediateSize;
    }

    ~BertLayer() {}

    float max_matrix(hpj::Matrix<float> &A) {
        float max = -9.0e9;

        for (int i = 0; i < A.Rows(); ++i) {
            float *presult = A.Row(i);
            for (int j = 0; j < A.Cols(); ++j) {
                if(fabs(presult[j]) > max){
                    max = fabs(presult[j]);
                }
            }
        }
        return max;
    }

    // FIXME(rfsaliev) rename `minmax` argument to avoid naming collision with `std::minmax`
    void setWeights(const float *_queryWeight, const float *_queryBias,
                    const float *_keyWeight, const float *_keyBias,
                    const float *_valueWeight, const float *_valueBias,
                    const float *_attentionOutputWeight, const float *_attentionOutputBias,
                    const float *_gamma1, const float *_beta1,
                    const float *_intermediateWeight, const float *_intermediateBias,
                    const float *_outputWeight, const float *_outputBias,
                    const float *_gamma2, const float *_beta2,
                    const float minmax[8]) {
        using namespace dnnl_wrappers;
        auto& eng = ctx.dnnl_context.getEngine();
        auto& stm = ctx.dnnl_context.getEngineStream();

        // query, key and value sizes are same
        auto m = maxTokenSize; // A.Rows();
        auto n = hiddenSize; // B.Cols();
        auto k = hiddenSize; // A.Cols() == B.Rows();

        // TODO(rfsaliev) use more structured form than float[] for minimum and maximum values
        qkv_SrcScale = computeQuantizationScale<input_t>(minmax[0], minmax[1]);

        std::tie(
            queryWeight,
            queryBias,
            queryMatMul_
        ) = BuildMatMul(m, n, k, qkv_SrcScale, _queryWeight, _queryBias);

        std::tie(
            keyWeight,
            keyBias,
            keyMatMul_
        ) = BuildMatMul(m, n, k, qkv_SrcScale, _keyWeight, _keyBias);

        std::tie(
            valueWeight,
            valueBias,
            valueMatMul_
        ) = BuildMatMul(m, n, k, qkv_SrcScale, _valueWeight, _valueBias);

        // Batch MatMul1 with bias and scale construction
        batchMatMul1ScaleBias_ = BuildBatchMatMul1WithScaleBias();

        // Softmax construction
        m = 12*maxTokenSize;
        n = maxTokenSize;
        const int axis = 1;
        softmax_ = std::make_unique<SoftMax>(MakeSoftmax<float>(eng, m, n, axis));

        // Batch MatMul2 construction
        batchMatMul2_ = BuildBatchMatMul2();

        // Weights for attention output
        m = maxTokenSize; // A.Rows();
        n = hiddenSize; // B.Cols();
        k = hiddenSize; // A.Cols() == B.Rows();

        attentionout_SrcScale = computeQuantizationScale<input_t>(minmax[2], minmax[3]);

        std::tie(
            attentionOutputWeight,
            attentionOutputBias,
            attentionMatMul_
        ) = BuildMatMul(m, n, k, attentionout_SrcScale, _attentionOutputWeight, _attentionOutputBias, BuildAttrs().Sum());

        // Batchnorm1
        const auto gamma1_mem = CloneMemory(eng, stm, {1, hiddenSize}, _gamma1);
        gamma1 = DataSource(gamma1_mem);
        const auto beta1_mem = CloneMemory(eng, stm, {1, hiddenSize}, _beta1);
        beta1 = DataSource(beta1_mem);

        m = maxTokenSize;
        n = hiddenSize;
        const float epsilon = 9.999999960041972e-13;
        const dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift;
        batchNorm_.reset(new LayerNorm(MakeLayerNorm<float>(eng, m, n, epsilon, flags)));

        // intermediate weight and bias
        m = maxTokenSize; // A.Rows();
        n = intermediateSize; // B.Cols();
        k = hiddenSize; // A.Cols() == B.Rows();

        intermediate_SrcScale = computeQuantizationScale<input_t>(minmax[4], minmax[5]);

        std::tie(
            intermediateWeight,
            intermediateBias,
            intermediateMatMul_
        ) = BuildMatMul(m, n, k, intermediate_SrcScale, _intermediateWeight, _intermediateBias, BuildAttrs().Eltwise(dnnl::algorithm::eltwise_gelu_erf));

        // output dense weight and bias
        m = maxTokenSize; // A.Rows();
        n = hiddenSize; // B.Cols();
        k = intermediateSize; // A.Cols() == B.Rows();

        out_SrcScale = computeQuantizationScale<input_t>(minmax[6], minmax[7]);

        std::tie(
            outputWeight,
            outputBias,
            outputMatMul_
        ) = BuildMatMul(m, n, k, out_SrcScale, _outputWeight, _outputBias, BuildAttrs().Sum());

        // Output batchnorm
        const auto gamma2_mem = CloneMemory(eng, stm, {1, hiddenSize}, _gamma2);
        gamma2 = DataSource(gamma2_mem);
        const auto beta2_mem = CloneMemory(eng, stm, {1, hiddenSize}, _beta2);
        beta2 = DataSource(beta2_mem);
    }

    // Do the forward computing for the whole BERT layer
    // input: maxTokenSize x hidden_size
    // actualTokens: #tokens = maxTokenSize - padded_tokens
    hpj::Matrix<float> &forward(hpj::Matrix<float> &inputBuffer) {

        using namespace dnnl_wrappers;
        auto& eng = ctx.dnnl_context.getEngine();
        auto& stm = ctx.dnnl_context.getEngineStream();

        auto inputBufferMem = AttachMemory(eng, inputBuffer);

        // TODO(rfsaliev) separate buffers for query, key and value
        hpj::Matrix<float> query(ctx.qkvMatMul, 0, ctx.maxTokenSize, 0, hiddenSize);
        hpj::Matrix<float> key(ctx.qkvMatMul, ctx.maxTokenSize, ctx.maxTokenSize, 0, hiddenSize);
        hpj::Matrix<float> value(ctx.qkvMatMul, 2*ctx.maxTokenSize, ctx.maxTokenSize, 0, hiddenSize);

#ifdef dynamic_quant
        qkv_SrcScale = computeQuantizationScale<input_t>(inputBuffer);
#endif

        auto qkv_SrcData = ScaledData(inputBufferMem, qkv_SrcScale);

        // Query
        auto queryMem = AttachMemory(eng, query);
        queryMatMul_->Compute(stm, qkv_SrcData, queryWeight, queryBias, queryMem);

        // Key
        auto keyMem = AttachMemory(eng, key);
        keyMatMul_->Compute(stm, qkv_SrcData, keyWeight, keyBias, keyMem);

        // Value
        auto valueMem = AttachMemory(eng, value);
        valueMatMul_->Compute(stm, qkv_SrcData, valueWeight, valueBias, valueMem);

        // Batch MatMul1 with bias and scale
        auto batchMatMul1_desc = batchMatMul1ScaleBias_->PrimDesc();
        auto batchMatMul1_QData = DataSource(AttachMemory(eng, query, batchMatMul1_desc.src_desc()));
        auto batchMatMul1_KData = DataSource(AttachMemory(eng, key, batchMatMul1_desc.weights_desc()));
        auto batchMatMul1_BData = DataSource(AttachMemory(eng, ctx.magic_value.get(), batchMatMul1_desc.bias_desc()));
        auto batchMatMul1_QKMem = AttachMemory(eng, ctx.qk_resultBuffer, batchMatMul1_desc.dst_desc());

        batchMatMul1ScaleBias_->Compute(stm, batchMatMul1_QData, batchMatMul1_KData, batchMatMul1_BData, batchMatMul1_QKMem);

        // Softmax
        auto qk_resultMem = AttachMemory(eng, {12*maxTokenSize, maxTokenSize}, ctx.qk_result[0]);
        auto qk_resultData = DataSource(qk_resultMem);
        softmax_->Compute(stm, qk_resultData, qk_resultMem);

        // Batch MatMul2
        auto batchMatMul2_desc = batchMatMul2_->PrimDesc();
        auto batchMatMul2_QKData = DataSource(AttachMemory(eng, ctx.qk_resultBuffer, batchMatMul2_desc.src_desc()));
        auto batchMatMul2_VData  = DataSource(AttachMemory(eng, value, batchMatMul2_desc.weights_desc()));
        auto batchMatMul2_BData  = DataSource();
        auto batchMatMul2_DMem = AttachMemory(eng, ctx.resultBuffer1, batchMatMul2_desc.dst_desc());

        batchMatMul2_->Compute(stm, batchMatMul2_QKData, batchMatMul2_VData, batchMatMul2_BData, batchMatMul2_DMem);

        // Attention Output
#ifdef dynamic_quant
        attentionout_SrcScale = computeQuantizationScale<input_t>(ctx.resultBuffer1);
#endif
        auto attentionOut_SrcData = ScaledData(AttachMemory(eng, ctx.resultBuffer1), attentionout_SrcScale);
        attentionMatMul_->Compute(stm, attentionOut_SrcData, attentionOutputWeight, attentionOutputBias, inputBufferMem);

        // BatchNorm 1
        auto inputBufferData = DataSource(inputBufferMem);
        batchNorm_->Compute(stm, inputBufferData, gamma1, beta1, inputBufferMem);

        // Intermediate with Erf
#ifdef dynamic_quant
        intermediate_SrcScale = computeQuantizationScale<input_t>(inputBuffer);
#endif
        auto intermediate_SrcData = ScaledData(inputBufferMem, intermediate_SrcScale);
        auto intermediateBufferMem = AttachMemory(eng, ctx.intermediateBuffer);
        intermediateMatMul_->Compute(stm, intermediate_SrcData, intermediateWeight, intermediateBias, intermediateBufferMem);

        // Output MatMul with Sum
#ifdef dynamic_quant
        // TODO(rfsaliev) analyze accuracy effect of the dynamic quantization here
        //                improve max_matrix() performance if dyn quant is unavoidable
        out_SrcScale = computeQuantizationScale<input_t>(ctx.intermediateBuffer);
#endif
        auto output_SrcData = ScaledData(intermediateBufferMem, out_SrcScale);

        outputMatMul_->Compute(stm, output_SrcData, outputWeight, outputBias, inputBufferMem);

        // Output batchnorm
        batchNorm_->Compute(stm, inputBufferData, gamma2, beta2, inputBufferMem);

        return inputBuffer;
    }

private:
    template <class T,
              typename = typename std::enable_if<
                                    std::is_arithmetic<T>::value
                                    || std::is_same<T, bfloat16>::value
                                  >::type>
    struct is_quantizable 
        : std::integral_constant<
            bool,
            std::is_integral<T>::value && !std::is_same<T, bfloat16>::value> {};

    template <class T>
    typename std::enable_if_t<!is_quantizable<T>::value, float>
    static constexpr computeQuantizationScale(...) {
        return dnnl_wrappers::BuildAttrs::noScale;
    }

    template <class T>
    typename std::enable_if_t<is_quantizable<T>::value, float>
    computeQuantizationScale(hpj::Matrix<float>& A) {
        return std::numeric_limits<T>::max() / max_matrix(A);
    }

    template <class T>
    typename std::enable_if_t<is_quantizable<T>::value, float>
    static constexpr computeQuantizationScale(const float* p, size_t size) {
        return std::numeric_limits<T>::max()
               / *std::max_element(p, p+size, [](const float& a, const float& b) { return std::abs(a) < std::abs(b); });
    }

    template <class T>
    typename std::enable_if_t<is_quantizable<T>::value, float>
    static constexpr computeQuantizationScale(float min, float max) {
        return std::numeric_limits<T>::max() / std::abs(max - min);
    }

    // Fake method just to test above templates
    void computeQScaleTest() {
        using namespace dnnl_wrappers;
        static_assert(computeQuantizationScale<float>(0.f, 254.f) == BuildAttrs::noScale);
        static_assert(computeQuantizationScale<bfloat16>(0.f, 254.f) == BuildAttrs::noScale);
        static_assert(computeQuantizationScale<int8_t>(0.f, 254.f) == .5f);
        static_assert(computeQuantizationScale<uint16_t>(0.f, 254.f) == BuildAttrs::noScale);
    }

    std::tuple<
        dnnl_wrappers::CachedDataSource,        // weight_data
        dnnl_wrappers::CachedDataSource,        // bias_data
        std::unique_ptr<dnnl_wrappers::MatMul>  // MatMul
    > BuildMatMul(int m, int n, int k, float src_scale, const float* weight, const float* bias, dnnl_wrappers::BuildAttrs attrs = {}) {
            using namespace dnnl_wrappers;
            auto& eng = ctx.dnnl_context.getEngine();
            auto& stm = ctx.dnnl_context.getEngineStream();

            const auto weight_scale = computeQuantizationScale<input_t>(weight, k * n);
            const auto bias_scale = src_scale * weight_scale;
            const auto dst_scale = 1.f / bias_scale;

            const auto weight_mem = CloneMemory(eng, stm, {k, n}, weight);
            const auto bias_mem = CloneMemory(eng, stm, {1, n}, bias);

            return std::make_tuple(
                    ScaledCachedData(weight_mem, weight_scale),
                    ScaledCachedData(bias_mem, bias_scale),
                    std::make_unique<MatMul>(MakeMatMul<input_t, input_t, float, float>(eng, m, n, k, attrs.Scale(dst_scale)))
            );
    }

#if BFLOAT16_ATTENTION
    using batch_input_t = bfloat16;
#else
    using batch_input_t = float;
#endif

    std::unique_ptr<dnnl_wrappers::MatMul> BuildBatchMatMul1WithScaleBias() {
        const int m = maxTokenSize;
        const int k = head_size;
        const int n = maxTokenSize; // B needs to transpose
        const int heads = hiddenSize / head_size;

        const auto s_dt = DnnlDataType<batch_input_t>::value;
        const auto w_dt = DnnlDataType<batch_input_t>::value;
        const auto b_dt = DnnlDataType<float>::value;
        const auto d_dt = DnnlDataType<float>::value;

        // B needs to transpose
        // dnnl::memory::format_tag::cab - is not defined
        const dnnl::memory::dims dnnl_strides__format_tag__cab{k, 1, heads * k};

        const dnnl::memory::desc     src_md{{heads, m, k}, s_dt, dnnl::memory::format_tag::bac};
        const dnnl::memory::desc weights_md{{heads, k, n}, w_dt, dnnl_strides__format_tag__cab};
        const dnnl::memory::desc    bias_md{{    1, 1, n}, b_dt, dnnl::memory::format_tag::abc};
        const dnnl::memory::desc     dst_md{{heads, m, n}, d_dt, dnnl::memory::format_tag::abc};

        const float scale = 0.125f;

        return std::make_unique<dnnl_wrappers::MatMul>(
                    ctx.dnnl_context.getEngine(),
                    src_md, weights_md, bias_md, dst_md,
                    dnnl_wrappers::BuildAttrs().Scale(scale));
    }

    std::unique_ptr<dnnl_wrappers::MatMul> BuildBatchMatMul2() {
        const int m = maxTokenSize;
        const int k = maxTokenSize;
        const int n = head_size;
        const int heads = hiddenSize / head_size;

        const auto s_dt = DnnlDataType<batch_input_t>::value;
        const auto w_dt = DnnlDataType<batch_input_t>::value;
        const auto d_dt = DnnlDataType<float>::value;

        const dnnl::memory::desc     src_md{{heads, m, k}, s_dt, dnnl::memory::format_tag::abc};
        const dnnl::memory::desc weights_md{{heads, k, n}, w_dt, dnnl::memory::format_tag::bac};
        const dnnl::memory::desc     dst_md{{heads, m, n}, d_dt, dnnl::memory::format_tag::bac};

        return std::make_unique<dnnl_wrappers::MatMul>(
                        ctx.dnnl_context.getEngine(),
                        src_md, weights_md, dnnl::memory::desc{}, dst_md,
                        dnnl::primitive_attr{});
    }

private:
    BertContext &ctx;
    int maxTokenSize;
    int hiddenSize;
    int intermediateSize;

    // Separate query, key, value weight, bias and MatMul op
    dnnl_wrappers::CachedDataSource queryWeight;
    dnnl_wrappers::CachedDataSource queryBias;
    std::unique_ptr<dnnl_wrappers::MatMul> queryMatMul_;

    std::unique_ptr<dnnl_wrappers::MatMul> batchMatMul1ScaleBias_;

    std::unique_ptr<dnnl_wrappers::SoftMax> softmax_;

    std::unique_ptr<dnnl_wrappers::MatMul> batchMatMul2_;

    dnnl_wrappers::CachedDataSource keyWeight;
    dnnl_wrappers::CachedDataSource keyBias;
    std::unique_ptr<dnnl_wrappers::MatMul> keyMatMul_;

    dnnl_wrappers::CachedDataSource valueWeight;
    dnnl_wrappers::CachedDataSource valueBias;
    std::unique_ptr<dnnl_wrappers::MatMul> valueMatMul_;

    dnnl_wrappers::CachedDataSource attentionOutputWeight;
    dnnl_wrappers::CachedDataSource attentionOutputBias;
    std::unique_ptr<dnnl_wrappers::MatMul> attentionMatMul_;

    dnnl_wrappers::DataSource gamma1;
    dnnl_wrappers::DataSource beta1;
    std::unique_ptr<dnnl_wrappers::LayerNorm> batchNorm_;

    dnnl_wrappers::CachedDataSource intermediateWeight;
    dnnl_wrappers::CachedDataSource intermediateBias;
    std::unique_ptr<dnnl_wrappers::MatMul> intermediateMatMul_;

    dnnl_wrappers::CachedDataSource outputWeight;
    dnnl_wrappers::CachedDataSource outputBias;
    std::unique_ptr<dnnl_wrappers::MatMul> outputMatMul_;

    dnnl_wrappers::DataSource gamma2;
    dnnl_wrappers::DataSource beta2;

    float qkv_SrcScale;
    float attentionout_SrcScale;
    float intermediate_SrcScale;
    float out_SrcScale;
};

#endif
