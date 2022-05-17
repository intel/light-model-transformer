// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_LAYER_H_
#define BERT_LAYER_H_

#include "dnnl_common.h"
#include "bert_context.h"
#include "dnnl_attr.hpp"
#include "dnnl_data.hpp"
#include "dnnl_ops.hpp"
#include "bert_type_traits.h"

#include "dnnl.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cassert>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

//#define dynamic_quant

namespace dnnl_wrappers {
/// Reinterpret memory keeping origin data_type
dnnl::memory reLayoutMemory(const dnnl::memory& mem, dnnl::memory::desc layout) {
    layout.data.data_type = mem.get_desc().data.data_type;
    assert(layout.get_size() == mem.get_desc().get_size());
    return dnnl::memory{layout, mem.get_engine(), mem.get_data_handle()};
}
} // namespace dnnl_wrappers

struct Layer_minmax {
    struct Min_max {
        float min;
        float max;
    };
    Min_max qkv;
    Min_max attentionout;
    Min_max intermediate;
    Min_max intermediate_post;
};

template <class BertContextT,
    class = std::enable_if_t<is_template_instance<BertContext, BertContextT>::value>
>
class BertLayer
{
    using batch_input_t = typename BertContextT::batch_input_t;
    using input_t = typename BertContextT::input_t;

public:
    static constexpr int head_size = BertContextT::head_size;

    // hiddenSize 768 Hidden layer neurons, number of hidden units
    // intermediateSize 3072 feed-forward/filter size dimension 4*hiddensize 
    BertLayer(BertContextT &_ctx, int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072)
    : ctx{_ctx}
    , batch_{ctx.batch_} {
        this->maxTokenSize = maxTokenSize;
        this->hiddenSize = hiddenSize;
        this->intermediateSize = intermediateSize;
    }

    ~BertLayer() {}

    // FIXME(rfsaliev) rename `minmax` argument to avoid naming collision with `std::minmax`
    void setWeights(const float *_queryWeight, const float *_queryBias,
                    const float *_keyWeight, const float *_keyBias,
                    const float *_valueWeight, const float *_valueBias,
                    const float *_attentionOutputWeight, const float *_attentionOutputBias,
                    const float *_gamma1, const float *_beta1,
                    const float *_intermediateWeight, const float *_intermediateBias,
                    const float *_outputWeight, const float *_outputBias,
                    const float *_gamma2, const float *_beta2,
                    const Layer_minmax& layer_minmax) {
        using namespace dnnl_wrappers;
        auto& eng = ctx.dnnl_context.getEngine();
        auto& stm = ctx.dnnl_context.getEngineStream();

        // query, key and value sizes are same
        auto m = maxTokenSize; // A.Rows();
        auto n = hiddenSize; // B.Cols();
        auto k = hiddenSize; // A.Cols() == B.Rows();

        qkv_SrcScale = computeQuantizationScale<input_t>(layer_minmax.qkv.min, layer_minmax.qkv.max);

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
        const auto qk_result_md = ctx.qk_resultBuffer.get_desc();
        const int axis = qk_result_md.dims().size() - 1;
        softmax_ = std::make_unique<SoftMax>(eng, qk_result_md, axis);

        // Batch MatMul2 construction
        batchMatMul2_ = BuildBatchMatMul2();

        // Weights for attention output
        m = maxTokenSize; // A.Rows();
        n = hiddenSize; // B.Cols();
        k = hiddenSize; // A.Cols() == B.Rows();

        attentionout_SrcScale = computeQuantizationScale<input_t>(layer_minmax.attentionout.min, layer_minmax.attentionout.max);

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
        batchNorm_ = std::make_unique<LayerNorm>(MakeLayerNorm<float>(eng, batch_, m, n, epsilon, flags));

        // intermediate weight and bias
        m = maxTokenSize; // A.Rows();
        n = intermediateSize; // B.Cols();
        k = hiddenSize; // A.Cols() == B.Rows();

        intermediate_SrcScale = computeQuantizationScale<input_t>(layer_minmax.intermediate.min,layer_minmax.intermediate.max);

        // Apply source quantization scale for output matmul to intermediate matmul result
        const auto intermediate_PostScale = computeQuantizationScale<input_t>(layer_minmax.intermediate_post.min, layer_minmax.intermediate_post.max);

        std::tie(
            intermediateWeight,
            intermediateBias,
            intermediateMatMul_
        ) = BuildMatMul<input_t, input_t, float, input_t>(m, n, k, intermediate_SrcScale, _intermediateWeight, _intermediateBias,
                        BuildAttrs().Eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f, intermediate_PostScale));

        // output dense weight and bias
        m = maxTokenSize; // A.Rows();
        n = hiddenSize; // B.Cols();
        k = intermediateSize; // A.Cols() == B.Rows();

        const auto out_WScale = computeQuantizationScale<input_t>(_outputWeight,  k * n);
        const auto out_BiasScale = intermediate_PostScale * out_WScale;

        std::tie(
            outputWeight,
            outputBias,
            outputMatMul_
        ) = BuildMatMul(m, n, k, out_WScale, out_BiasScale, _outputWeight, _outputBias, BuildAttrs().Sum());

        // Output batchnorm
        const auto gamma2_mem = CloneMemory(eng, stm, {1, hiddenSize}, _gamma2);
        gamma2 = DataSource(gamma2_mem);
        const auto beta2_mem = CloneMemory(eng, stm, {1, hiddenSize}, _beta2);
        beta2 = DataSource(beta2_mem);
    }

    // Do the forward computing for the whole BERT layer
    // input: maxTokenSize x hidden_size
    // actualTokens: #tokens = maxTokenSize - padded_tokens
    void forward(dnnl::memory& inputBufferMem) {

        assert([&](){
            auto dims = inputBufferMem.get_desc().dims();
            return dims.size() == 3 && dims[0] == batch_ && dims[1] == maxTokenSize && dims[2] == hiddenSize;
        }());

        using namespace dnnl_wrappers;
        auto& stm = ctx.dnnl_context.getEngineStream();

#ifdef dynamic_quant
        qkv_SrcScale = computeQuantizationScale<input_t>(inputBufferMem);
#endif

        auto qkv_SrcData = ScaledData(inputBufferMem, qkv_SrcScale);

        // Query
        queryMatMul_->Compute(stm, qkv_SrcData, queryWeight, queryBias, ctx.query);

        // Key
        keyMatMul_->Compute(stm, qkv_SrcData, keyWeight, keyBias, ctx.key);

        // Value
        valueMatMul_->Compute(stm, qkv_SrcData, valueWeight, valueBias, ctx.value);


        // Batch MatMul1 with bias and scale
        auto batchMatMul1_desc = batchMatMul1ScaleBias_->PrimDesc();
        auto batchMatMul1_QData = DataSource(reLayoutMemory(ctx.query, batchMatMul1_desc.src_desc()));
        auto batchMatMul1_KData = DataSource(reLayoutMemory(ctx.key, batchMatMul1_desc.weights_desc()));
        auto batchMatMul1_MaskData = DataSource(ctx.input_mask);

        std::unordered_map<int, std::reference_wrapper<DataSource>> post_ops_data = {{0, std::ref(batchMatMul1_MaskData)}};
        batchMatMul1ScaleBias_->ComputeWithPostOps(stm, batchMatMul1_QData, batchMatMul1_KData, post_ops_data,
                                                ctx.qk_resultBuffer);

        // Softmax
        auto qk_resultData = DataSource(ctx.qk_resultBuffer);
        softmax_->Compute(stm, qk_resultData, ctx.qk_resultBuffer);

        // Batch MatMul2
        auto batchMatMul2_desc = batchMatMul2_->PrimDesc();
        auto batchMatMul2_QKData = DataSource(ctx.qk_resultBuffer);
        auto batchMatMul2_VData  = DataSource(reLayoutMemory(ctx.value, batchMatMul2_desc.weights_desc()));
        auto batchMatMul2_BData  = DataSource();
        auto batchMatMul2_DMem = reLayoutMemory(ctx.resultBuffer1, batchMatMul2_desc.dst_desc());

        batchMatMul2_->Compute(stm, batchMatMul2_QKData, batchMatMul2_VData, batchMatMul2_BData, batchMatMul2_DMem);

        // Attention Output
#ifdef dynamic_quant
        attentionout_SrcScale = computeQuantizationScale<input_t>(ctx.resultBuffer1);
#endif
        auto attentionOut_SrcData = ScaledData(ctx.resultBuffer1, attentionout_SrcScale);
        attentionMatMul_->Compute(stm, attentionOut_SrcData, attentionOutputWeight, attentionOutputBias, inputBufferMem);

        // BatchNorm 1
        auto inputBufferData = DataSource(inputBufferMem);
        batchNorm_->Compute(stm, inputBufferData, gamma1, beta1, inputBufferMem);

        // Intermediate with Erf
#ifdef dynamic_quant
        intermediate_SrcScale = computeQuantizationScale<input_t>(inputBufferMem);
#endif
        auto intermediate_SrcData = ScaledData(inputBufferMem, intermediate_SrcScale);
        intermediateMatMul_->Compute(stm, intermediate_SrcData, intermediateWeight, intermediateBias, ctx.intermediateBuffer);

        // Output MatMul with Sum
        auto output_SrcData = DataSource(ctx.intermediateBuffer);

        outputMatMul_->Compute(stm, output_SrcData, outputWeight, outputBias, inputBufferMem);

        // Output batchnorm
        batchNorm_->Compute(stm, inputBufferData, gamma2, beta2, inputBufferMem);
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
    static constexpr computeQuantizationScale(const float* p, size_t size) {
        return std::numeric_limits<T>::max()
               / *std::max_element(p, p+size, [](const float& a, const float& b) { return std::abs(a) < std::abs(b); });
    }

    template <class T>
    typename std::enable_if_t<is_quantizable<T>::value, float>
    static constexpr computeQuantizationScale(float min, float max) {
        return std::numeric_limits<T>::max() / std::abs(max - min);
    }

    // For dynamic quantization case
    // TODO(rfsaliev) use dnnl::reduction to compute max value
    template <class T>
    typename std::enable_if_t<is_quantizable<T>::value, float>
    computeQuantizationScale(const dnnl::memory& mem) {
        assert(mem.get_desc().data_type() == dnnl::memory::data_type::f32);
        ctx.dnnl_context.getEngineStream().wait();
        return computeQuantizationScale<T>(MemoryAccessor<float>(mem).Data(), mem.get_desc().get_size() / sizeof(float));
    }

    // Fake method just to test above templates
    void computeQScaleTest() {
        using namespace dnnl_wrappers;
        static_assert(computeQuantizationScale<float>(0.f, 254.f) == BuildAttrs::noScale, "");
        static_assert(computeQuantizationScale<bfloat16>(0.f, 254.f) == BuildAttrs::noScale, "");
        static_assert(computeQuantizationScale<int8_t>(0.f, 254.f) == .5f, "");
        static_assert(computeQuantizationScale<uint16_t>(0.f, 254.f) == BuildAttrs::noScale, "");
    }

    template <class Src_T = input_t, class Weight_T = Src_T, class Bias_T = float, class Dst_T = float>
    std::tuple<
        dnnl_wrappers::CachedDataSource,        // weight_data
        dnnl_wrappers::CachedDataSource,        // bias_data
        std::unique_ptr<dnnl_wrappers::MatMul>  // MatMul
    > BuildMatMul(int m, int n, int k,
                  float weight_scale, float bias_scale,
                  const float* weight, const float* bias,
                  dnnl_wrappers::BuildAttrs attrs = {}) {
            using namespace dnnl_wrappers;
            auto& eng = ctx.dnnl_context.getEngine();
            auto& stm = ctx.dnnl_context.getEngineStream();

            const auto dst_scale = 1.f / bias_scale;

            const auto weight_mem = CloneMemory(eng, stm, {1, k, n}, weight);
            const auto bias_mem = CloneMemory(eng, stm, {1, 1, n}, bias);

            return std::make_tuple(
                    ScaledCachedData(weight_mem, weight_scale),
                    ScaledCachedData(bias_mem, bias_scale),
                    std::make_unique<MatMul>(MakeMatMul<Src_T, Weight_T, Bias_T, Dst_T>(eng, batch_, m, n, k, attrs.Scale(dst_scale)))
            );
    }

    template <class Src_T = input_t, class Weight_T = Src_T, class Bias_T = float, class Dst_T = float>
    std::tuple<
        dnnl_wrappers::CachedDataSource,        // weight_data
        dnnl_wrappers::CachedDataSource,        // bias_data
        std::unique_ptr<dnnl_wrappers::MatMul>  // MatMul
    > BuildMatMul(int m, int n, int k, float src_scale, const float* weight, const float* bias, dnnl_wrappers::BuildAttrs attrs = {}) {
            const auto weight_scale = computeQuantizationScale<Weight_T>(weight, k * n);
            const auto bias_scale = src_scale * weight_scale;
            return BuildMatMul<Src_T, Weight_T, Bias_T, Dst_T>(m, n, k, weight_scale, bias_scale, weight, bias, attrs);
    }

    std::unique_ptr<dnnl_wrappers::MatMul> BuildBatchMatMul1WithScaleBias() {
        const int m = maxTokenSize;
        const int k = head_size;
        const int n = maxTokenSize; // B needs to transpose
        const int heads = hiddenSize / head_size;

        const auto s_dt = DnnlDataType<batch_input_t>::value;
        const auto w_dt = DnnlDataType<batch_input_t>::value;
        const auto d_dt = DnnlDataType<float>::value;

        // B needs to transpose
        // dnnl::memory::format_tag::cab - is not defined
        const dnnl::memory::dims dnnl_strides__format_tag__adbc{heads * k * m, k, 1, heads * k};

        const dnnl::memory::desc     src_md{{batch_, heads, m, k}, s_dt, dnnl::memory::format_tag::acbd};
        const dnnl::memory::desc weights_md{{batch_, heads, k, n}, w_dt, dnnl_strides__format_tag__adbc};
        const dnnl::memory::desc    bias_md{};
        const dnnl::memory::desc     dst_md{{batch_, heads, m, n}, d_dt, dnnl::memory::format_tag::abcd};
        
        const float scale = 0.125f;

        return std::make_unique<dnnl_wrappers::MatMul>(
                    ctx.dnnl_context.getEngine(),
                    src_md, weights_md, bias_md, dst_md,
                    dnnl_wrappers::BuildAttrs()
                        .Scale(scale)
                        .Binary(dnnl::algorithm::binary_add, ctx.input_mask.get_desc())
                        );
    }

    std::unique_ptr<dnnl_wrappers::MatMul> BuildBatchMatMul2() {
        const int m = maxTokenSize;
        const int k = maxTokenSize;
        const int n = head_size;
        const int heads = hiddenSize / head_size;

        const auto s_dt = DnnlDataType<batch_input_t>::value;
        const auto w_dt = DnnlDataType<batch_input_t>::value;
        const auto d_dt = DnnlDataType<float>::value;

        const dnnl::memory::desc     src_md{{batch_, heads, m, k}, s_dt, dnnl::memory::format_tag::abcd};
        const dnnl::memory::desc weights_md{{batch_, heads, k, n}, w_dt, dnnl::memory::format_tag::acbd};
        const dnnl::memory::desc     dst_md{{batch_, heads, m, n}, d_dt, dnnl::memory::format_tag::acbd};

        return std::make_unique<dnnl_wrappers::MatMul>(
                        ctx.dnnl_context.getEngine(),
                        src_md, weights_md, dnnl::memory::desc{}, dst_md,
                        dnnl::primitive_attr{});
    }

private:
    BertContextT &ctx;
    int maxTokenSize;
    int hiddenSize;
    int intermediateSize;
    int batch_;

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
};

#endif
