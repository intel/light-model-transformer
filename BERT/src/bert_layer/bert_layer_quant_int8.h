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
    static constexpr float outputQuantizationAccuracyFactor = 3.f;

    // ctx->hiddenSize 768 Hidden layer neurons, number of hidden units
    // ctx->intermediateSize 3072 feed-forward/filter size dimension 4*ctx->hiddenSize 
    BertLayer(const std::shared_ptr<BertContext> &_ctx)
    : ctx{_ctx}
    , intermediateBufType{ctx->UnsignedQuantizationType()} {
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

        // Default operation types
        const auto src_type    = ctx->UnsignedQuantizationType();
        const auto weight_type = ctx->SignedQuantizationType();
        const auto bias_type   = ctx->use_quantization ? dt::s32 : ctx->FloatType();
        const auto dst_type    = ctx->FloatType();
        const auto op_data_types = OpDataTypes{src_type, weight_type, bias_type, dst_type};

        auto& eng = ctx->dnnl_context.getEngine();

        // query, key and value sizes are same
        auto m = ctx->maxTokenSize; // A.Rows();
        auto n = ctx->hiddenSize; // B.Cols();
        auto k = ctx->hiddenSize; // A.Cols() == B.Rows();
        queryIPDesc = BuildInnerProduct(op_data_types, m, n, k, quant_factors_.qkv, _queryWeight, _queryBias);
        keyIPDesc   = BuildInnerProduct(op_data_types, m, n, k, quant_factors_.qkv,   _keyWeight,   _keyBias);
        valueIPDesc = BuildInnerProduct(op_data_types, m, n, k, quant_factors_.qkv, _valueWeight, _valueBias);

        // Batch MatMul1 with bias and scale construction
        batchMatMul1ScaleBias_ = BuildBatchMatMul1WithScaleBias();

        // Softmax construction
        const auto qk_result_md = ctx->qk_resultBuffer.get_desc();
        const int axis = qk_result_md.dims().size() - 1;
        softmax_ = std::make_unique<SoftMax>(eng, qk_result_md, axis);

        // Batch MatMul2 construction
        batchMatMul2_ = BuildBatchMatMul2();

        // Attention Output MatMul construction
        m = ctx->maxTokenSize; // A.Rows();
        n = ctx->hiddenSize; // B.Cols();
        k = ctx->hiddenSize; // A.Cols() == B.Rows();
        attentionOutIPDesc = BuildInnerProduct(op_data_types,
                                               m, n, k,
                                               quant_factors_.attention_out,
                                               _attentionOutputWeight, _attentionOutputBias,
                                               BuildAttrs().Sum());

        // Norm1
        const auto gamma1_mem = ReshapeMemory(_gamma1, {1, ctx->hiddenSize});
        gamma1 = DataSource(gamma1_mem);
        const auto beta1_mem = ReshapeMemory(_beta1, {1, ctx->hiddenSize});
        beta1 = DataSource(beta1_mem);

        auto ln1_md = attentionOutIPDesc.prim->PrimDesc().dst_desc();
        const float epsilon = 9.999999960041972e-13;
        const dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift;
        Norm1_ = std::make_unique<LayerNorm>(eng, ln1_md, epsilon, flags);

        // - At first, construct Output MatMul - its scale/shift will be used later

        // Compute intermediate buffer type depending on accuracy factor
        intermediateBufType = src_type;
        auto outputWeightType  = weight_type;
        auto outputBiasType = bias_type;

        // Force float output MatMul if scaling less than accuracy factor
        const auto output_SrcScale = computeQuantizationScale(intermediateBufType, quant_factors_.output.min, quant_factors_.output.max);
        if (output_SrcScale < outputQuantizationAccuracyFactor) {
            intermediateBufType = outputWeightType = outputBiasType = ctx->FloatType();
        }

        // output dense weight and bias
        m = ctx->maxTokenSize; // A.Rows();
        n = ctx->hiddenSize; // B.Cols();
        k = ctx->intermediateSize; // A.Cols() == B.Rows();
        outputIPDesc = BuildInnerProduct({intermediateBufType, outputWeightType, outputBiasType, dst_type},
                                         m, n, k,
                                         _quant_factors.output,
                                         _outputWeight,
                                         _outputBias,
                                         BuildAttrs().Sum());

        // - At second, construct Intermediate MatMul

        // Apply source quantization scale of the Output MatMul to Intermediate MatMul result
        using algo = dnnl::algorithm;
        auto intermediateAttrs = BuildAttrs().Eltwise(algo::eltwise_gelu, 0.f, 0.f, outputIPDesc.scale);

        if (outputIPDesc.shift != 0.f) {
            intermediateAttrs.Eltwise(algo::eltwise_linear, 1.f, outputIPDesc.shift);
        }

        // intermediate weight and bias
        m = ctx->maxTokenSize; // A.Rows();
        n = ctx->intermediateSize; // B.Cols();
        k = ctx->hiddenSize; // A.Cols() == B.Rows();
        intermediateIPDesc = BuildInnerProduct({src_type, weight_type, bias_type, intermediateBufType},
                                                m, n, k,
                                                _quant_factors.intermediate,
                                                _intermediateWeight, _intermediateBias,
                                                intermediateAttrs);

        // Output Norm
        const auto gamma2_mem = ReshapeMemory(_gamma2, {1, ctx->hiddenSize});
        gamma2 = DataSource(gamma2_mem);
        const auto beta2_mem = ReshapeMemory(_beta2, {1, ctx->hiddenSize});
        beta2 = DataSource(beta2_mem);

        auto ln2_md = outputIPDesc.prim->PrimDesc().dst_desc();
        Norm2_ = std::make_unique<LayerNorm>(eng, ln2_md, epsilon, flags);
    }

    // Do the forward computing for the whole BERT layer
    // input: ctx->maxTokenSize x hidden_size
    // actualTokens: #tokens = ctx->maxTokenSize - padded_tokens
    void forward(dnnl::memory inputBufferMem, const dnnl::memory& input_mask) {
        using namespace dnnl_wrappers;
        auto& stm = ctx->dnnl_context.getEngineStream();

        dnnl::memory::dims input_dims = {ctx->batch_ * ctx->maxTokenSize, ctx->hiddenSize};
        inputBufferMem = ReshapeMemory(inputBufferMem, input_dims);

        if (ctx->calibrate_quant_factors)
        {
            quant_factors_.qkv.Update(inputBufferMem);
        }

        auto qkv_SrcData = queryIPDesc.ScaledCachedData(inputBufferMem);

        // Query
        queryIPDesc.Compute(stm, qkv_SrcData, ctx->query);

        // Key
        keyIPDesc.Compute(stm, qkv_SrcData, ctx->key);

        // Value
        valueIPDesc.Compute(stm, qkv_SrcData, ctx->value);


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
        auto attentionOut_SrcData = attentionOutIPDesc.ScaledData(ctx->resultBuffer1);
        attentionOutIPDesc.Compute(stm, attentionOut_SrcData, inputBufferMem);

        // Norm 1
        auto inputBufferData = DataSource(inputBufferMem);
        Norm1_->Compute(stm, inputBufferData, gamma1, beta1, inputBufferMem);

        // Intermediate with Erf
        auto intermediate_SrcData = intermediateIPDesc.ScaledData(inputBufferMem);
        intermediateIPDesc.Compute(stm, intermediate_SrcData, ctx->intermediateBuffer(intermediateBufType));

        if (ctx->calibrate_quant_factors)
        {
            quant_factors_.attention_out.Update(ctx->resultBuffer1);
            quant_factors_.intermediate.Update(inputBufferMem);
            quant_factors_.output.Update(ctx->intermediateBuffer(intermediateBufType));
        }

        // Output MatMul with Sum
        auto output_SrcData = DataSource(ctx->intermediateBuffer(intermediateBufType));
        outputIPDesc.Compute(stm, output_SrcData, inputBufferMem);

        // Output Norm
        Norm2_->Compute(stm, inputBufferData, gamma2, beta2, inputBufferMem);
    }

    const QuantizationFactors& QuantFactors()
    {
        return quant_factors_;
    }

private:
    // Shift bias for source zero-point case with formula:
    // let: weight[N, K], bias[N]
    // bias[n] -= reduce_sum(weight, K)[n] * zero_point
    dnnl::memory shiftBias(int n, int k, dnnl_wrappers::DataSource& weight, dnnl_wrappers::DataSource& bias, float zero_point) {
        using namespace dnnl_wrappers;
        using reduce = dnnl::reduction;
        using algo = dnnl::algorithm;
        using dt = dnnl::memory::data_type;

        const auto weight_quant_type = ctx->SignedQuantizationType();
        auto& stm = ctx->dnnl_context.getEngineStream();

        if (zero_point == 0.f || weight_quant_type == ctx->FloatType()) {
            return bias.GetData(stm, {{n}, ctx->FloatType(), dnnl::memory::dims{}});
        }

        auto src = weight.GetData(stm, {{n, k}, weight_quant_type, dnnl::memory::dims{}});

        auto bias_mem = bias.GetData(stm, {{n}, dt::s32, dnnl::memory::dims{}});
        auto dst = ReshapeMemory(bias_mem, {n, 1});

        reduce prim{
            reduce::primitive_desc{
                reduce::desc{algo::reduction_sum, src.get_desc(), dst.get_desc(), 0.f, 0.f},
                BuildAttrs().Eltwise(algo::eltwise_linear, -zero_point).Sum(),
                ctx->dnnl_context.getEngine()
            }
        };

        prim.execute(
            stm,
            {
                {DNNL_ARG_SRC, src},
                {DNNL_ARG_DST, dst},
            }
        );
        stm.wait();

        return bias_mem;
    }

    struct InnerProductInferenceDesc {
        dnnl_wrappers::CachedDataSource weight;
        dnnl_wrappers::CachedDataSource bias;
        std::unique_ptr<dnnl_wrappers::InnerProduct> prim;
        float scale = 1.f;
        float shift = 0.f;
        // dnnl::memory is actually shared pointer to a buffer
        // there is the chance that context will change underlying buffer for scratchpad space
        // so let's store a pointer to context's scratchpad memory field rather than copy underlying buffer
        // this will allow us to re-use 1 scratchpad buffer for all primitives
        std::shared_ptr<dnnl::memory> scratchpad;

        void Compute(dnnl::stream& stm, dnnl_wrappers::DataSource& src, dnnl::memory& dst_memory) {
            prim->Compute(stm, src, weight, bias, dst_memory, scratchpad ? *scratchpad : dnnl::memory{});
        }

        dnnl_wrappers::DataSource ScaledData(const dnnl::memory& mem) {
            return dnnl_wrappers::ScaledData(mem, scale, shift);
        }

        dnnl_wrappers::CachedDataSource ScaledCachedData(const dnnl::memory& mem) {
            return dnnl_wrappers::ScaledCachedData(mem, scale, shift);
        }
    };

    InnerProductInferenceDesc BuildInnerProduct(const OpDataTypes& data_types,
                                                int m, int n, int k,
                                                const MinMax& min_max,
                                                const dnnl::memory& weight,
                                                const dnnl::memory& bias,
                                                dnnl_wrappers::BuildAttrs attrs = {}) {
            using namespace dnnl_wrappers;
            InnerProductInferenceDesc ipConfig;

            ipConfig.scale = computeQuantizationScale(data_types.src_dt, min_max.min, min_max.max);
            // TODO(rfsaliev) std::round(), std::ceil() or std::floor()?
            ipConfig.shift = (data_types.src_dt == dnnl::memory::data_type::u8)
                ? std::round(-min_max.min * ipConfig.scale)
                : 0.f;

            auto& eng = ctx->dnnl_context.getEngine();

            const auto weight_scale = computeQuantizationScale(data_types.weight_dt, weight, ctx->dnnl_context.getEngineStream());
            const auto bias_scale = ipConfig.scale * weight_scale;
            const auto dst_scale = 1.f / bias_scale;

            // Note(rfsaliev): InnerProduct expects memory format 'OI' which is transposition to Matmul 'KN'
            // Note(krzychut): The AttachMemory call is only safe if the lifetime of the buffer can be guaranteed
            // by outside code.
            const auto weight_mem = AttachMemory(eng, {n, k}, (float*)weight.get_data_handle(), true);
            const auto bias_mem = ReshapeMemory(bias, {n});

            ipConfig.weight = ScaledCachedData(weight_mem, weight_scale);
            auto scaled_bias = ScaledData(bias_mem, bias_scale);
            ipConfig.bias = CachedDataSource(shiftBias(n, k, ipConfig.weight, scaled_bias, ipConfig.shift));

            ipConfig.prim   = std::make_unique<InnerProduct>(MakeInnerProduct(
                                       eng, ctx->batch_, m, n, k, data_types.src_dt, data_types.weight_dt,
                                       data_types.bias_dt, data_types.dst_dt, attrs.Scale(dst_scale).ScratchpadModeUser()));

            ipConfig.scratchpad = ctx->AllocateScratchpad(ipConfig.prim->PrimDesc().scratchpad_desc());

            return ipConfig;
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
        const auto d_dt = ctx->qk_resultBuffer.get_desc().data_type();

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
        const auto d_dt = ctx->resultBuffer1.get_desc().data_type();

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
    InnerProductInferenceDesc queryIPDesc;
    InnerProductInferenceDesc keyIPDesc;
    InnerProductInferenceDesc valueIPDesc;

    std::unique_ptr<dnnl_wrappers::MatMul> batchMatMul1ScaleBias_;

    std::unique_ptr<dnnl_wrappers::SoftMax> softmax_;

    std::unique_ptr<dnnl_wrappers::MatMul> batchMatMul2_;

    InnerProductInferenceDesc attentionOutIPDesc;

    dnnl_wrappers::DataSource gamma1;
    dnnl_wrappers::DataSource beta1;
    std::unique_ptr<dnnl_wrappers::LayerNorm> Norm1_;

    InnerProductInferenceDesc intermediateIPDesc;
    InnerProductInferenceDesc outputIPDesc;

    dnnl_wrappers::DataSource gamma2;
    dnnl_wrappers::DataSource beta2;
    std::unique_ptr<dnnl_wrappers::LayerNorm> Norm2_;

    dnnl::memory::data_type intermediateBufType;

    QuantizationFactors quant_factors_;
};

#endif
