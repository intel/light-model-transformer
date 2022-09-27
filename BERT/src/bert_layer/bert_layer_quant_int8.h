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
#include "quant_factors.hpp"

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


class BertLayer
{
    using dt = dnnl::memory::data_type;

    struct OpDataTypes {
        dt src_dt;
        dt weight_dt;
        dt bias_dt;
        dt dst_dt;
    };

public:
    static constexpr int head_size = BertContext::head_size;

    // ctx->hiddenSize 768 Hidden layer neurons, number of hidden units
    // ctx->intermediateSize 3072 feed-forward/filter size dimension 4*ctx->hiddenSize 
    BertLayer(const std::shared_ptr<BertContext> &_ctx)
    : ctx{_ctx} {
    }

    ~BertLayer() {}

    void setWeights(const dnnl::memory& _queryWeight, const dnnl::memory& _queryBias,
                    const dnnl::memory& _keyWeight, const dnnl::memory& _keyBias,
                    const dnnl::memory& _valueWeight, const dnnl::memory& _valueBias,
                    const dnnl::memory& _attentionOutputWeight, const dnnl::memory& _attentionOutputBias,
                    const dnnl::memory& _gamma1, const dnnl::memory& _beta1,
                    const dnnl::memory& _intermediateWeight, const dnnl::memory& _intermediateBias,
                    const dnnl::memory& _outputWeight, const dnnl::memory& _outputBias,
                    const dnnl::memory& _gamma2, const dnnl::memory& _beta2,
                    const QuantizationFactors& _quant_factors = QuantizationFactors{}) {
        using namespace dnnl_wrappers;
        quant_factors_ = _quant_factors;

        auto& eng = ctx->dnnl_context.getEngine();

        // query, key and value sizes are same
        auto m = ctx->maxTokenSize; // A.Rows();
        auto n = ctx->hiddenSize; // B.Cols();
        auto k = ctx->hiddenSize; // A.Cols() == B.Rows();

        const auto quantization_type = ctx->QuantizationType();

        qkv_SrcScale = computeQuantizationScale(quantization_type, quant_factors_.qkv.min, quant_factors_.qkv.max);

        OpDataTypes op_data_types{quantization_type, quantization_type, dt::f32, dt::f32};
        std::tie(
            queryWeight,
            queryBias,
            queryMatMul_
        ) = BuildInnerProduct(op_data_types, m, n, k, qkv_SrcScale, _queryWeight, _queryBias);

        std::tie(
            keyWeight,
            keyBias,
            keyMatMul_
        ) = BuildInnerProduct(op_data_types, m, n, k, qkv_SrcScale, _keyWeight, _keyBias);

        std::tie(
            valueWeight,
            valueBias,
            valueMatMul_
        ) = BuildInnerProduct(op_data_types, m, n, k, qkv_SrcScale, _valueWeight, _valueBias);

        // Batch MatMul1 with bias and scale construction
        batchMatMul1ScaleBias_ = BuildBatchMatMul1WithScaleBias();

        // Softmax construction
        const auto qk_result_md = ctx->qk_resultBuffer.get_desc();
        const int axis = qk_result_md.dims().size() - 1;
        softmax_ = std::make_unique<SoftMax>(eng, qk_result_md, axis);

        // Batch MatMul2 construction
        batchMatMul2_ = BuildBatchMatMul2();

        // Weights for attention output
        m = ctx->maxTokenSize; // A.Rows();
        n = ctx->hiddenSize; // B.Cols();
        k = ctx->hiddenSize; // A.Cols() == B.Rows();

        attentionout_SrcScale = computeQuantizationScale(quantization_type, quant_factors_.attention_out.min, quant_factors_.attention_out.max);

        std::tie(
            attentionOutputWeight,
            attentionOutputBias,
            attentionMatMul_
        ) = BuildInnerProduct(op_data_types, m, n, k, attentionout_SrcScale, _attentionOutputWeight, _attentionOutputBias, BuildAttrs().Sum());

        // Norm1
        const auto gamma1_mem = ReshapeMemory(_gamma1, {1, ctx->hiddenSize});
        gamma1 = DataSource(gamma1_mem);
        const auto beta1_mem = ReshapeMemory(_beta1, {1, ctx->hiddenSize});
        beta1 = DataSource(beta1_mem);

        auto ln1_md = attentionMatMul_->PrimDesc().dst_desc();
        const float epsilon = 9.999999960041972e-13;
        const dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift;
        Norm1_ = std::make_unique<LayerNorm>(eng, ln1_md, epsilon, flags);

        // intermediate weight and bias
        m = ctx->maxTokenSize; // A.Rows();
        n = ctx->intermediateSize; // B.Cols();
        k = ctx->hiddenSize; // A.Cols() == B.Rows();

        intermediate_SrcScale = computeQuantizationScale(quantization_type, quant_factors_.intermediate.min,quant_factors_.intermediate.max);

        // Apply source quantization scale for output matmul to intermediate matmul result
        const auto intermediate_PostScale = computeQuantizationScale(quantization_type, quant_factors_.intermediate_post.min, quant_factors_.intermediate_post.max);

        op_data_types = {quantization_type, quantization_type, dt::f32, quantization_type};
        std::tie(
            intermediateWeight,
            intermediateBias,
            intermediateMatMul_
        ) = BuildInnerProduct(op_data_types, m, n, k, intermediate_SrcScale, _intermediateWeight, _intermediateBias,
                        BuildAttrs().Eltwise(dnnl::algorithm::eltwise_gelu_erf, 0.f, 0.f, intermediate_PostScale));

        // output dense weight and bias
        m = ctx->maxTokenSize; // A.Rows();
        n = ctx->hiddenSize; // B.Cols();
        k = ctx->intermediateSize; // A.Cols() == B.Rows();

        const auto out_WScale = computeQuantizationScale(quantization_type, _outputWeight);
        const auto out_BiasScale = intermediate_PostScale * out_WScale;

        op_data_types = {quantization_type, quantization_type, dt::f32, dt::f32};
        std::tie(
            outputWeight,
            outputBias,
            outputMatMul_
        ) = BuildInnerProduct(op_data_types, m, n, k, out_WScale, out_BiasScale, _outputWeight, _outputBias, BuildAttrs().Sum());

        // Output Norm
        const auto gamma2_mem = ReshapeMemory(_gamma2, {1, ctx->hiddenSize});
        gamma2 = DataSource(gamma2_mem);
        const auto beta2_mem = ReshapeMemory(_beta2, {1, ctx->hiddenSize});
        beta2 = DataSource(beta2_mem);

        auto ln2_md = outputMatMul_->PrimDesc().dst_desc();
        Norm2_ = std::make_unique<LayerNorm>(eng, ln2_md, epsilon, flags);
    }

    // Do the forward computing for the whole BERT layer
    // input: ctx->maxTokenSize x hidden_size
    // actualTokens: #tokens = ctx->maxTokenSize - padded_tokens
    void forward(dnnl::memory& inputBufferMem, const dnnl::memory& input_mask) {
        using namespace dnnl_wrappers;
        auto& stm = ctx->dnnl_context.getEngineStream();

        dnnl::memory::dims input_dims = {ctx->batch_ * ctx->maxTokenSize, ctx->hiddenSize};
        inputBufferMem = ReshapeMemory(inputBufferMem, input_dims);

        if (ctx->calibrate_quant_factors)
        {
            quant_factors_.qkv.Update(inputBufferMem);
        }

        auto qkv_SrcData = ScaledData(inputBufferMem, qkv_SrcScale);

        // Query
        queryMatMul_->Compute(stm, qkv_SrcData, queryWeight, queryBias, ctx->query);

        // Key
        keyMatMul_->Compute(stm, qkv_SrcData, keyWeight, keyBias, ctx->key);

        // Value
        valueMatMul_->Compute(stm, qkv_SrcData, valueWeight, valueBias, ctx->value);


        // Batch MatMul1 with bias and scale
        auto reshaped_input_mask = ReshapeMemory(input_mask, MaskDescriptor().dims());
        auto batchMatMul1_desc = batchMatMul1ScaleBias_->PrimDesc();
        auto batchMatMul1_QData = DataSource(ReLayoutMemory(ctx->query, batchMatMul1_desc.src_desc()));
        auto batchMatMul1_KData = DataSource(ReLayoutMemory(ctx->key, batchMatMul1_desc.weights_desc()));
        auto batchMatMul1_MaskData = DataSource(reshaped_input_mask);

        std::unordered_map<int, std::reference_wrapper<DataSource>> post_ops_data = {{0, std::ref(batchMatMul1_MaskData)}};
        batchMatMul1ScaleBias_->ComputeWithPostOps(stm, batchMatMul1_QData, batchMatMul1_KData, post_ops_data,
                                                ctx->qk_resultBuffer);

        // Softmax
        auto qk_resultData = DataSource(ctx->qk_resultBuffer);
        softmax_->Compute(stm, qk_resultData, ctx->qk_resultBuffer);

        // Batch MatMul2
        auto batchMatMul2_desc = batchMatMul2_->PrimDesc();
        auto batchMatMul2_QKData = DataSource(ctx->qk_resultBuffer);
        auto batchMatMul2_VData  = DataSource(ReLayoutMemory(ctx->value, batchMatMul2_desc.weights_desc()));
        auto batchMatMul2_BData  = DataSource();
        auto batchMatMul2_DMem = ReLayoutMemory(ctx->resultBuffer1, batchMatMul2_desc.dst_desc());

        batchMatMul2_->Compute(stm, batchMatMul2_QKData, batchMatMul2_VData, batchMatMul2_BData, batchMatMul2_DMem);

        // Attention Output
        auto attentionOut_SrcData = ScaledData(ctx->resultBuffer1, attentionout_SrcScale);
        attentionMatMul_->Compute(stm, attentionOut_SrcData, attentionOutputWeight, attentionOutputBias, inputBufferMem);

        // Norm 1
        auto inputBufferData = DataSource(inputBufferMem);
        Norm1_->Compute(stm, inputBufferData, gamma1, beta1, inputBufferMem);

        // Intermediate with Erf
        auto intermediate_SrcData = ScaledData(inputBufferMem, intermediate_SrcScale);
        intermediateMatMul_->Compute(stm, intermediate_SrcData, intermediateWeight, intermediateBias, ctx->intermediateBuffer);

        if (ctx->calibrate_quant_factors)
        {
            quant_factors_.attention_out.Update(ctx->resultBuffer1);
            quant_factors_.intermediate.Update(inputBufferMem);
            quant_factors_.intermediate_post.Update(ctx->intermediateBuffer);
        }

        // Output MatMul with Sum
        auto output_SrcData = DataSource(ctx->intermediateBuffer);

        outputMatMul_->Compute(stm, output_SrcData, outputWeight, outputBias, inputBufferMem);

        // Output Norm
        Norm2_->Compute(stm, inputBufferData, gamma2, beta2, inputBufferMem);
    }

    const QuantizationFactors& QuantFactors()
    {
        return quant_factors_;
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
            std::is_integral<T>::value> {};

    template <class... Args>
    float computeQuantizationScale(dt data_type, Args&&... args)
    {
        switch(data_type)
        {
            case dt::s8:
                return computeQuantizationScale<int8_t>(std::forward<Args>(args)...);
            case dt::u8:
                return computeQuantizationScale<uint8_t>(std::forward<Args>(args)...);
            case dt::s32:
                return computeQuantizationScale<int32_t>(std::forward<Args>(args)...);
            default:
                return dnnl_wrappers::BuildAttrs::noScale;
        }
    }

    template <class T>
    typename std::enable_if_t<is_quantizable<T>::value, float>
    static constexpr computeQuantizationScale(const float* p, size_t size) {
        // std::max_element is not constexpr until c++17
        return static_cast<float>(std::numeric_limits<T>::max())
               / *std::max_element(p, p+size, [](const float& a, const float& b) { return std::abs(a) < std::abs(b); });
    }

    template <class T>
    typename std::enable_if_t<is_quantizable<T>::value, float>
    static constexpr computeQuantizationScale(float min, float max) {
        return static_cast<float>(std::numeric_limits<T>::max()) / std::abs(max - min);
        // TODO: (krzychut) clarify if smaller range would be enough:
        // return static_cast<float>(std::numeric_limits<T>::max()) / std::max(std::abs(min), std::abs(max));
    }

    // For dynamic quantization case
    // TODO(rfsaliev) use dnnl::reduction to compute max value
    template <class T>
    typename std::enable_if_t<is_quantizable<T>::value, float>
    computeQuantizationScale(const dnnl::memory& mem) {
        assert(mem.get_desc().data_type() == dnnl::memory::data_type::f32);
        ctx->dnnl_context.getEngineStream().wait();
        return computeQuantizationScale<T>(MemoryAccessor<float>(mem).Data(), mem.get_desc().get_size() / sizeof(float));
    }

    // Fake method just to test above templates
    void computeQScaleTest() {
        using namespace dnnl_wrappers;
        static_assert(computeQuantizationScale<int8_t>(0.f, 254.f) == .5f, "");
    }



    std::tuple<
        dnnl_wrappers::CachedDataSource,        // weight_data
        dnnl_wrappers::CachedDataSource,        // bias_data
        std::unique_ptr<dnnl_wrappers::MatMul>  // MatMul
    > BuildMatMul(OpDataTypes data_types, int m, int n, int k,
                  float weight_scale, float bias_scale,
                  const dnnl::memory& weight, const dnnl::memory& bias,
                  dnnl_wrappers::BuildAttrs attrs = {}) {
            using namespace dnnl_wrappers;
            auto& eng = ctx->dnnl_context.getEngine();

            const auto dst_scale = 1.f / bias_scale;

            const auto weight_mem = ReshapeMemory(weight, {1, k, n});
            const auto bias_mem = ReshapeMemory(bias, {1, 1, n});

            return std::make_tuple(
                ScaledCachedData(weight_mem, weight_scale), ScaledCachedData(bias_mem, bias_scale),
                std::make_unique<MatMul>(MakeMatMul(eng, ctx->batch_, m, n, k, data_types.src_dt, data_types.weight_dt,
                                                    data_types.bias_dt, data_types.dst_dt, attrs.Scale(dst_scale))));
    }

    std::tuple<
        dnnl_wrappers::CachedDataSource,        // weight_data
        dnnl_wrappers::CachedDataSource,        // bias_data
        std::unique_ptr<dnnl_wrappers::MatMul>  // MatMul
    > BuildMatMul(OpDataTypes data_types, int m, int n, int k, float src_scale, const dnnl::memory& weight, const dnnl::memory& bias, dnnl_wrappers::BuildAttrs attrs = {}) {
            const auto weight_scale = computeQuantizationScale(data_types.weight_dt, weight);
            const auto bias_scale = src_scale * weight_scale;
            return BuildMatMul(data_types, m, n, k, weight_scale, bias_scale, weight, bias, attrs);
    }

    std::tuple<
        dnnl_wrappers::CachedDataSource,        // weight_data
        dnnl_wrappers::CachedDataSource,        // bias_data
        std::unique_ptr<dnnl_wrappers::InnerProduct>  // InnerProduct
    > BuildInnerProduct(OpDataTypes data_types, int m, int n, int k,
                  float weight_scale, float bias_scale,
                  const dnnl::memory& weight, const dnnl::memory& bias,
                  dnnl_wrappers::BuildAttrs attrs = {}) {
            using namespace dnnl_wrappers;
            auto& eng = ctx->dnnl_context.getEngine();

            const auto dst_scale = 1.f / bias_scale;

            // Note(rfsaliev): InnerProduct expects memory format 'OI' which is transposition to Matmul 'KN'
            // Note(krzychut): The AttachMemory call is only safe if the lifetime of the buffer can be guaranteed
            // by outside code.
            const auto weight_mem = AttachMemory(eng, {n, k}, (float*)weight.get_data_handle(), true);
            const auto bias_mem = ReshapeMemory(bias, {n});

            return std::make_tuple(ScaledCachedData(weight_mem, weight_scale), ScaledCachedData(bias_mem, bias_scale),
                                   std::make_unique<InnerProduct>(MakeInnerProduct(
                                       eng, ctx->batch_, m, n, k, data_types.src_dt, data_types.weight_dt,
                                       data_types.bias_dt, data_types.dst_dt, attrs.Scale(dst_scale))));
    }

    std::tuple<
        dnnl_wrappers::CachedDataSource,        // weight_data
        dnnl_wrappers::CachedDataSource,        // bias_data
        std::unique_ptr<dnnl_wrappers::InnerProduct>  // MatMul
    > BuildInnerProduct(OpDataTypes data_types, int m, int n, int k, float src_scale, const dnnl::memory& weight, const dnnl::memory& bias, dnnl_wrappers::BuildAttrs attrs = {}) {
            const auto weight_scale = computeQuantizationScale(data_types.weight_dt , weight);
            const auto bias_scale = src_scale * weight_scale;
            return BuildInnerProduct(data_types, m, n, k, weight_scale, bias_scale, weight, bias, attrs);
    }

    std::unique_ptr<dnnl_wrappers::MatMul> BuildBatchMatMul1WithScaleBias() {
        const int m = ctx->maxTokenSize;
        const int k = head_size;
        const int n = ctx->maxTokenSize; // B needs to transpose
        const int batch = ctx->batch_;
        const int heads = ctx->numHeads;

        const auto float_dtype = ctx->FloatType();
        const auto s_dt = float_dtype;
        const auto w_dt = float_dtype;
        const auto d_dt = DnnlDataType<float>::value;

        // B needs to transpose
        // dnnl::memory::format_tag::cab - is not defined
        // const dnnl::memory::dims dnnl_strides__format_tag__adbc{heads * k * m, k, 1, heads * k};

        const dnnl::memory::desc     src_md{{batch, heads, m, k}, s_dt, dnnl::memory::format_tag::acbd};
        const dnnl::memory::desc weights_md{{batch, heads, k, n}, w_dt, dnnl::memory::format_tag::adbc};
        const dnnl::memory::desc    bias_md{};
        const dnnl::memory::desc     dst_md{{batch, heads, m, n}, d_dt, dnnl::memory::format_tag::abcd};

        const dnnl::memory::desc    mask_md{{ctx->batch_, 1, 1, ctx->maxTokenSize}, dt::f32, dnnl::memory::dims{}};

        const float scale = 0.125f;
        return std::make_unique<dnnl_wrappers::MatMul>(
                    ctx->dnnl_context.getEngine(),
                    src_md, weights_md, bias_md, dst_md,
                    dnnl_wrappers::BuildAttrs()
                        .Scale(scale)
                        .Binary(dnnl::algorithm::binary_add, mask_md)
                        );
    }

    std::unique_ptr<dnnl_wrappers::MatMul> BuildBatchMatMul2() {
        const int m = ctx->maxTokenSize;
        const int k = ctx->maxTokenSize;
        const int n = head_size;
        const int heads = ctx->numHeads;
        const int batch = ctx->batch_;

        const auto float_dtype = ctx->FloatType();
        const auto s_dt = float_dtype;
        const auto w_dt = float_dtype;
        const auto d_dt = DnnlDataType<float>::value;

        const dnnl::memory::desc     src_md{{batch, heads, m, k}, s_dt, dnnl::memory::format_tag::abcd};
        const dnnl::memory::desc weights_md{{batch, heads, k, n}, w_dt, dnnl::memory::format_tag::acbd};
        const dnnl::memory::desc     dst_md{{batch, heads, m, n}, d_dt, dnnl::memory::format_tag::acbd};

        return std::make_unique<dnnl_wrappers::MatMul>(
                        ctx->dnnl_context.getEngine(),
                        src_md, weights_md, dnnl::memory::desc{}, dst_md,
                        dnnl::primitive_attr{});
    }

    dnnl::memory::desc MaskDescriptor()
    {
        const auto prim_desc = batchMatMul1ScaleBias_->PrimDesc();
        auto attr = prim_desc.get_primitive_attr();
        auto post_ops = attr.get_post_ops();
        dnnl::algorithm alg;
        dnnl::memory::desc desc;
        post_ops.get_params_binary(0, alg, desc);
        return desc;
    }
private:
    std::shared_ptr<BertContext> ctx;

    // Separate query, key, value weight, bias and MatMul op
    dnnl_wrappers::CachedDataSource queryWeight;
    dnnl_wrappers::CachedDataSource queryBias;
    std::unique_ptr<dnnl_wrappers::InnerProduct> queryMatMul_;

    std::unique_ptr<dnnl_wrappers::MatMul> batchMatMul1ScaleBias_;

    std::unique_ptr<dnnl_wrappers::SoftMax> softmax_;

    std::unique_ptr<dnnl_wrappers::MatMul> batchMatMul2_;

    dnnl_wrappers::CachedDataSource keyWeight;
    dnnl_wrappers::CachedDataSource keyBias;
    std::unique_ptr<dnnl_wrappers::InnerProduct> keyMatMul_;

    dnnl_wrappers::CachedDataSource valueWeight;
    dnnl_wrappers::CachedDataSource valueBias;
    std::unique_ptr<dnnl_wrappers::InnerProduct> valueMatMul_;

    dnnl_wrappers::CachedDataSource attentionOutputWeight;
    dnnl_wrappers::CachedDataSource attentionOutputBias;
    std::unique_ptr<dnnl_wrappers::InnerProduct> attentionMatMul_;

    dnnl_wrappers::DataSource gamma1;
    dnnl_wrappers::DataSource beta1;
    std::unique_ptr<dnnl_wrappers::LayerNorm> Norm1_;

    dnnl_wrappers::CachedDataSource intermediateWeight;
    dnnl_wrappers::CachedDataSource intermediateBias;
    std::unique_ptr<dnnl_wrappers::InnerProduct> intermediateMatMul_;

    dnnl_wrappers::CachedDataSource outputWeight;
    dnnl_wrappers::CachedDataSource outputBias;
    std::unique_ptr<dnnl_wrappers::InnerProduct> outputMatMul_;

    dnnl_wrappers::DataSource gamma2;
    dnnl_wrappers::DataSource beta2;
    std::unique_ptr<dnnl_wrappers::LayerNorm> Norm2_;

    float qkv_SrcScale;
    float attentionout_SrcScale;
    float intermediate_SrcScale;

    QuantizationFactors quant_factors_;
};

#endif
