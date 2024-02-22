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
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>
#include <unordered_map>


class BertLayer
{
    using InnerProductPrimT = dnnl::inner_product_forward;

    static dnnl::memory::data_type DataQuantizationType(const std::shared_ptr<BertContext>& ctx) {
        return dnnl_wrappers::AsymQuantizationSupported<InnerProductPrimT>()
            ? ctx->UnsignedQuantizationType()
            : ctx->SignedQuantizationType();
    }

    using dt = dnnl::memory::data_type;

    struct OpDataTypes {
        dt src_dt;
        dt weight_dt;
        dt bias_dt;
        dt dst_dt;
    };

    enum class Ops {
        query,
        key,
        value,
        batchMatMul1,
        softmax,
        batchMatMul2,
        attentionOut,
        norm1,
        intermediate,
        output,
        norm2
    };

    static const std::map<Ops, std::string>& OpsToNames() {
        #define BERT_OP_PAIR(x) {Ops::x, #x}
        static const std::map<Ops, std::string> opsToNames {
            BERT_OP_PAIR(query),
            BERT_OP_PAIR(key),
            BERT_OP_PAIR(value),
            BERT_OP_PAIR(batchMatMul1),
            BERT_OP_PAIR(softmax),
            BERT_OP_PAIR(batchMatMul2),
            BERT_OP_PAIR(attentionOut),
            BERT_OP_PAIR(norm1),
            BERT_OP_PAIR(intermediate),
            BERT_OP_PAIR(output),
            BERT_OP_PAIR(norm2)
        };
        #undef BERT_OP_PAIR
        return opsToNames;
    }

public:
    static constexpr float outputQuantizationAccuracyFactor = 3.f;

    static std::vector<std::string> OpNames() {
        static auto& opsToNames = OpsToNames();

        std::vector<std::string> result;
        result.reserve(opsToNames.size());

        std::transform(
            std::begin(opsToNames),
            std::end(opsToNames),
            std::back_inserter(result),
            [](const auto& pair) { return pair.second; });

        return result;
    }

    // ctx->hiddenSize 768 Hidden layer neurons, number of hidden units
    // ctx->intermediateSize 3072 feed-forward/filter size dimension 4*ctx->hiddenSize 
    BertLayer(const std::shared_ptr<BertContext> &_ctx)
    : ctx{_ctx}
    , intermediateBufType{DataQuantizationType(ctx)} {
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
        const auto src_type    = DataQuantizationType(ctx);
        const auto weight_type = ctx->SignedQuantizationType();
        const auto bias_type   = ctx->FloatType();
        const auto dst_type    = ctx->FloatType();
        const auto op_data_types = OpDataTypes{src_type, weight_type, bias_type, dst_type};

        auto& eng = ctx->dnnl_context.getEngine();

        queryKeyValue_ = std::make_unique<QueryKeyValue>(ctx,
                                      op_data_types,
                                     _queryWeight, _queryBias,
                                     _keyWeight, _keyBias,
                                     _valueWeight, _valueBias,
                                     quant_factors_.qkv);

        // Batch MatMul1 with bias and scale construction
        batchMatMul1ScaleBias_ = std::make_unique<BatchMatMul1WithScaleBias>(ctx, queryKeyValue_->getQueryMD(), queryKeyValue_->getKeyMD());

        // Softmax construction
        const auto qk_result_md = batchMatMul1ScaleBias_->dst_desc();
        const int axis = qk_result_md.get_dims().size() - 1;
        softmax_ = std::make_unique<SoftMax>(eng, qk_result_md, axis);

        // At first, construct att out to use src_desc in batchMatMul2
        // Attention Output MatMul construction
        auto m = ctx->maxTokenSize; // A.Rows();
        auto n = ctx->hiddenSize; // B.Cols();
        auto k = ctx->hiddenSize; // A.Cols() == B.Rows();
        attentionOutIPDesc = BuildInnerProduct(ctx, op_data_types,
                                               m, n, k,
                                               quant_factors_.attention_out,
                                               _attentionOutputWeight, _attentionOutputBias,
                                               BuildAttrs().Sum());

        // Batch MatMul2 construction
        batchMatMul2_ = std::make_unique<BatchMatMul2>(ctx, queryKeyValue_->getValueMD(), attentionOutIPDesc.dst_desc());

        // Norm1
        const auto gamma1_mem = ReshapeMemory(_gamma1, {1, ctx->hiddenSize});
        gamma1 = ImmutableDataSource(gamma1_mem);
        const auto beta1_mem = ReshapeMemory(_beta1, {1, ctx->hiddenSize});
        beta1 = ImmutableDataSource(beta1_mem);

        auto ln1_md = ConvertIPDataDims(attentionOutIPDesc.prim->PrimDesc().dst_desc(), 2);
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
        if (output_SrcScale > 1.f / outputQuantizationAccuracyFactor) {
            intermediateBufType = outputWeightType = outputBiasType = ctx->FloatType();
        }

        // output dense weight and bias
        m = ctx->maxTokenSize; // A.Rows();
        n = ctx->hiddenSize; // B.Cols();
        k = ctx->intermediateSize; // A.Cols() == B.Rows();
        outputIPDesc = BuildInnerProduct(ctx, {intermediateBufType, outputWeightType, outputBiasType, dst_type},
                                         m, n, k,
                                         _quant_factors.output,
                                         _outputWeight,
                                         _outputBias,
                                         BuildAttrs().Sum());

        // - At second, construct Intermediate MatMul

        // Apply source quantization scale of the Output MatMul to Intermediate MatMul result
        using algo = dnnl::algorithm;
        auto intermediateAttrs = BuildAttrs()
                                    .Eltwise(algo::eltwise_gelu_tanh)
                                    .Scale(outputIPDesc.scale, DNNL_ARG_DST)
                                    .ZeroPoint(outputIPDesc.shift, DNNL_ARG_DST);


        // intermediate weight and bias
        m = ctx->maxTokenSize; // A.Rows();
        n = ctx->intermediateSize; // B.Cols();
        k = ctx->hiddenSize; // A.Cols() == B.Rows();
        intermediateIPDesc = BuildInnerProduct(ctx, {src_type, weight_type, bias_type, intermediateBufType},
                                                m, n, k,
                                                _quant_factors.intermediate,
                                                _intermediateWeight, _intermediateBias,
                                                intermediateAttrs);

        // Output Norm
        const auto gamma2_mem = ReshapeMemory(_gamma2, {1, ctx->hiddenSize});
        gamma2 = ImmutableDataSource(gamma2_mem);
        const auto beta2_mem = ReshapeMemory(_beta2, {1, ctx->hiddenSize});
        beta2 = ImmutableDataSource(beta2_mem);

        auto ln2_md = ConvertIPDataDims(outputIPDesc.prim->PrimDesc().dst_desc(), 2);
        Norm2_ = std::make_unique<LayerNorm>(eng, ln2_md, epsilon, flags);
    }

    // Do the forward computing for the whole BERT layer
    // input: ctx->maxTokenSize x hidden_size
    // actualTokens: #tokens = ctx->maxTokenSize - padded_tokens
    void forward(dnnl::memory& inputBufferMem, const dnnl::memory& input_mask) {
        using namespace dnnl_wrappers;

        if (inputBufferMem.get_desc() != ResultMD()) {
            throw std::runtime_error("BertLayer: input memory descriptor does not match");
        }

        auto& stm = ctx->dnnl_context.getEngineStream();
        static auto& opsToNames = OpsToNames();

        if (ctx->calibrate_quant_factors)
        {
            quant_factors_.qkv.Update(inputBufferMem);
        }

        // Query+key+Value
        auto qkv_resultBuffer = ctx->PopBuffer(queryKeyValue_->getResultMD());
        dnnl::memory query_mem, key_mem, value_mem;
        std::tie(query_mem, key_mem, value_mem) = queryKeyValue_->Compute(inputBufferMem, qkv_resultBuffer);

        // Batch MatMul1 with bias and scale
        auto qk_resultBuffer = ctx->PopBuffer(batchMatMul1ScaleBias_->dst_desc());
        ctx->profiler.Profile(opsToNames.at(Ops::batchMatMul1), [&](){
            batchMatMul1ScaleBias_->Compute(stm, query_mem, key_mem, input_mask, qk_resultBuffer);
        });

        // Softmax
        auto qk_resultData = ImmutableDataSource(qk_resultBuffer);
        ctx->profiler.Profile(opsToNames.at(Ops::softmax), [&](){
            softmax_->Compute(stm, qk_resultData, qk_resultBuffer);
        });

        // Batch MatMul2
        auto resultBuffer1 = ctx->PopBuffer(attentionOutIPDesc.dst_desc());
        ctx->profiler.Profile(opsToNames.at(Ops::batchMatMul2), [&](){
            batchMatMul2_->Compute(stm, qk_resultBuffer, value_mem, resultBuffer1);
        });

        // Attention Output
        auto attentionOut_SrcData = attentionOutIPDesc.ScaledData(resultBuffer1);
        ctx->profiler.Profile(opsToNames.at(Ops::attentionOut), [&](){
            attentionOutIPDesc.Compute(stm, attentionOut_SrcData, inputBufferMem);
        });

        // Norm 1
        auto norm1Buffer = ReLayoutMemory(inputBufferMem, Norm1_->PrimDesc().dst_desc());
        auto norm1BufferData = ImmutableDataSource(norm1Buffer);
        ctx->profiler.Profile(opsToNames.at(Ops::norm1), [&](){
            Norm1_->Compute(stm, norm1BufferData, gamma1, beta1, norm1Buffer);
        });

        // Intermediate with Erf
        auto intermediate_SrcData = intermediateIPDesc.ScaledData(inputBufferMem);
        auto intermediateBuffer = ctx->PopBuffer(intermediateIPDesc.dst_desc());
        ctx->profiler.Profile(opsToNames.at(Ops::intermediate), [&](){
            intermediateIPDesc.Compute(stm, intermediate_SrcData, intermediateBuffer);
        });

        if (ctx->calibrate_quant_factors)
        {
            quant_factors_.attention_out.Update(resultBuffer1);
            quant_factors_.intermediate.Update(inputBufferMem);
            quant_factors_.output.Update(intermediateBuffer);
        }

        // Output MatMul with Sum
        auto output_SrcData = ImmutableDataSource(intermediateBuffer);
        ctx->profiler.Profile(opsToNames.at(Ops::output), [&](){
            outputIPDesc.Compute(stm, output_SrcData, inputBufferMem);
        });

        // Output Norm
        auto norm2Buffer = ReLayoutMemory(inputBufferMem, Norm2_->PrimDesc().dst_desc());
        auto norm2BufferData = ImmutableDataSource(norm2Buffer);
        ctx->profiler.Profile(opsToNames.at(Ops::norm2), [&](){
            Norm2_->Compute(stm, norm2BufferData, gamma2, beta2, norm2Buffer);
        });
    }

    const QuantizationFactors& QuantFactors()
    {
        return quant_factors_;
    }

    dnnl::memory::desc ResultMD() const {
        if (!attentionOutIPDesc.prim) {
            throw std::logic_error("BertLayer is not initialized");
        }
        return attentionOutIPDesc.dst_desc();
    }

    dnnl::memory PrepareInput(const dnnl::memory& input) const {
        auto result_md = ResultMD();
        auto input_md = input.get_desc();
        if (result_md == input_md) {
            return input;
        }

        assert((input_md.get_dims() == dnnl::memory::dims{ctx->batch_ * ctx->maxTokenSize, ctx->hiddenSize}
             || input_md.get_dims() == dnnl::memory::dims{ctx->batch_,  ctx->maxTokenSize, ctx->hiddenSize}));

        // join batch and maxTokenSize dimensions
        auto reshaped_input_md = input_md.get_ndims() == 2
                                 ? input_md
                                 : input_md.reshape({static_cast<dnnl::memory::dim>(ctx->batch_) * ctx->maxTokenSize, ctx->hiddenSize});
        // reinterpret to match result_md dimensions
        using namespace dnnl_wrappers;
        reshaped_input_md = ConvertIPDataDims(reshaped_input_md, result_md.get_ndims());
        auto reshaped_input = ReLayoutMemory(input, reshaped_input_md);

        if (reshaped_input_md == result_md)
            return reshaped_input;

        dnnl::memory result{result_md, ctx->dnnl_context.getEngine()};

        auto& stm = ctx->dnnl_context.getEngineStream();
        dnnl::reorder{reshaped_input, result}.execute(stm, reshaped_input, result);
        stm.wait();
        return result;
    }

    void ProcessResult(dnnl::memory& result, dnnl::memory& output) const {
        assert(result.get_desc() == ResultMD());
        if (result.get_data_handle() == output.get_data_handle())
            return;

        auto result_md = result.get_desc();
        auto output_md = output.get_desc();

        assert((output_md.get_dims() == dnnl::memory::dims{ctx->batch_ * ctx->maxTokenSize, ctx->hiddenSize}
             || output_md.get_dims() == dnnl::memory::dims{ctx->batch_,  ctx->maxTokenSize, ctx->hiddenSize}));

        // join batch and maxTokenSize dimensions
        auto reshaped_output_md = output_md.get_ndims() == 2 
                                ? output_md
                                : output_md.reshape({static_cast<dnnl::memory::dim>(ctx->batch_) * ctx->maxTokenSize, ctx->hiddenSize});
        // reinterpret to match result_md dimensions
        using namespace dnnl_wrappers;
        auto reshaped_output = ReLayoutMemory(output, ConvertIPDataDims(reshaped_output_md, result_md.get_ndims()));

        auto& stm = ctx->dnnl_context.getEngineStream();
        dnnl::reorder{result, reshaped_output}.execute(stm, result, reshaped_output);
        stm.wait();
    }

private:
    struct InnerProductInferenceDesc {
        dnnl_wrappers::CachedDataSource weight;
        dnnl_wrappers::CachedDataSource bias;
        std::unique_ptr<dnnl_wrappers::InnerProduct<InnerProductPrimT>> prim;
        dnnl::memory scale;
        dnnl::memory shift;
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

        dnnl::memory::desc dst_desc() const {
            if (dst_desc_.is_zero()) {
                dst_desc_ = prim->PrimDesc().dst_desc();
            }
            return dst_desc_;
        }
    private:
        mutable dnnl::memory::desc dst_desc_{};
    };

    static InnerProductInferenceDesc BuildInnerProduct(const std::shared_ptr<BertContext>& ctx,
                                                const OpDataTypes& data_types,
                                                int m, int n, int k,
                                                const MinMax& min_max,
                                                const dnnl::memory& weight,
                                                const dnnl::memory& bias,
                                                dnnl_wrappers::BuildAttrs attrs = {}) {
            using namespace dnnl_wrappers;
            auto& eng = ctx->dnnl_context.getEngine();
            auto& stm = ctx->dnnl_context.getEngineStream();

            InnerProductInferenceDesc ipConfig;
            const auto scale_val = computeQuantizationScale(data_types.src_dt, min_max.min, min_max.max);
            if (scale_val != BuildAttrs::noScale) {
                ipConfig.scale = ToMemory(eng, stm, scale_val);
            }
            // TODO(rfsaliev) std::round(), std::ceil() or std::floor()?
            if (data_types.src_dt == dnnl::memory::data_type::u8) {
                ipConfig.shift = ToMemory(eng, stm, static_cast<int32_t>(std::round(-min_max.min / scale_val)));
            }

            const auto weight_scale = computeQuantizationScale(data_types.weight_dt, weight, ctx->dnnl_context.getEngineStream());

            // Note(rfsaliev): InnerProduct expects memory format 'OI' which is transposition to Matmul 'KN'
            // Note(krzychut): The AttachMemory call is only safe if the lifetime of the buffer can be guaranteed
            // by outside code.
            const auto dims = MakeInnerProductDims<InnerProductPrimT>(ctx->batch_, m , n, k);
            const auto w_fmt = std::vector<dnnl::memory::format_tag>{
                dnnl::memory::format_tag::undef, // ndims == 0
                dnnl::memory::format_tag::a,     // ndims == 1
                dnnl::memory::format_tag::ba,    // == 2
                dnnl::memory::format_tag::bac,   // 3
                dnnl::memory::format_tag::bacd,  // 4
                dnnl::memory::format_tag::bacde  // 5
            }.at(dims.weights_tz.size());

            const auto weight_mem = ReLayoutMemory(weight, dnnl::memory::desc{dims.weights_tz, dt::f32, w_fmt});
            const auto bias_mem = ReshapeMemory(bias, dims.bias_tz);

            const auto weight_scale_mem = weight_scale != BuildAttrs::noScale
                                            ? ToMemory(eng, stm, weight_scale)
                                            : dnnl::memory{};
            ipConfig.weight = ScaledCachedData(weight_mem, weight_scale_mem);

            ipConfig.bias = CachedDataSource(bias_mem);

            attrs.Scale(ipConfig.scale, DNNL_ARG_SRC).ZeroPoint(ipConfig.shift, DNNL_ARG_SRC)
                 .Scale(weight_scale_mem, DNNL_ARG_WEIGHTS);

            ipConfig.prim   = std::make_unique<InnerProduct<InnerProductPrimT>>(MakeInnerProduct<InnerProductPrimT>(
                                       eng, ctx->batch_, m, n, k, data_types.src_dt, data_types.weight_dt,
                                       data_types.bias_dt, data_types.dst_dt, attrs.ScratchpadModeUser()));

            ipConfig.scratchpad = ctx->AllocateScratchpad(ipConfig.prim->PrimDesc().scratchpad_desc());

            return ipConfig;
    }

    class QueryKeyValue {
    public:
        QueryKeyValue(const std::shared_ptr<BertContext>& _ctx,
                      const OpDataTypes& op_data_types,
                      const dnnl::memory& _queryWeight, const dnnl::memory& _queryBias,
                      const dnnl::memory& _keyWeight, const dnnl::memory& _keyBias,
                      const dnnl::memory& _valueWeight, const dnnl::memory& _valueBias, const MinMax& qkv_min_max)
            : ctx{_ctx} {

            using namespace dnnl_wrappers;
            auto& eng = ctx->dnnl_context.getEngine();

            // query, key and value sizes are same
            auto m = ctx->maxTokenSize; // A.Rows();
            auto n = ctx->hiddenSize; // B.Cols();
            auto k = ctx->hiddenSize; // A.Cols() == B.Rows();

            // weights: io {k,n}
            // bias: {n}

            if (joinQKV()) {

                qkvWeight = joinData({k, n}, {_queryWeight, _keyWeight, _valueWeight});
                assert(qkvWeight.get_desc().get_dims() == (dnnl::memory::dims{k, n*3}));

                qkvBias = joinData({n}, {_queryBias, _keyBias, _valueBias});
                assert(qkvBias.get_desc().get_dims() == (dnnl::memory::dims{n*3}));

                qkvIPDesc = BertLayer::BuildInnerProduct(ctx, op_data_types, m, n*3, k, qkv_min_max, qkvWeight, qkvBias);

                const auto qkv_dst_md = ConvertIPDataDims(qkvIPDesc.dst_desc(), 2);
                auto qkv_dims = qkv_dst_md.get_dims();
                assert(qkv_dims == (dnnl::memory::dims{ctx->batch_ * m, n*3}));

                auto qkv_reshape_md = qkv_dst_md.reshape({qkv_dims[0]/m, qkv_dims[0]/ctx->batch_, 3, qkv_dims[1]/3}).permute_axes({1,2,0,3});
                qkvResultMD = {qkv_reshape_md.get_dims(), qkv_reshape_md.get_data_type(), dnnl::memory::dims{}};
                qkvReorder = {dnnl::reorder::primitive_desc{eng, qkv_reshape_md, eng, qkvResultMD}};

                qkv_dims = qkvResultMD.get_dims();
                qkv_dims[0] = 1;
                auto qkv_part_md = dnnl::memory::desc{qkv_dims, qkvResultMD.get_data_type(), dnnl::memory::dims{}}.reshape({qkv_dims[1] * qkv_dims[2], qkv_dims[3]});

                queryMD = keyMD = valueMD = qkv_part_md;

            } else /*!joinQKV*/ {

                queryIPDesc = BertLayer::BuildInnerProduct(ctx, op_data_types, m, n, k, qkv_min_max, _queryWeight, _queryBias);
                  keyIPDesc = BertLayer::BuildInnerProduct(ctx, op_data_types, m, n, k, qkv_min_max,   _keyWeight,   _keyBias);
                valueIPDesc = BertLayer::BuildInnerProduct(ctx, op_data_types, m, n, k, qkv_min_max, _valueWeight, _valueBias);

                queryMD = queryIPDesc.dst_desc();
                  keyMD =   keyIPDesc.dst_desc();
                valueMD = valueIPDesc.dst_desc();

                qkvResultMD = dnnl::memory::desc{
                    dnnl::memory::dims{static_cast<dnnl::memory::dim>(
                        queryMD.get_size() + keyMD.get_size() + valueMD.get_size()
                    )},
                    dnnl::memory::data_type::u8,
                    dnnl::memory::dims{}
                };
            }

            offsetQuery = 0;
            offsetKey   = offsetQuery + queryMD.get_size();
            offsetValue = offsetKey   +   keyMD.get_size();
        }

        std::tuple<dnnl::memory, dnnl::memory, dnnl::memory>
        Compute(dnnl::memory& inputBufferMem, const BertContext::BufferHandler& qkvResultBuffer) {
            assert(qkvResultBuffer.get().get_desc() == qkvResultMD);

            auto& eng = ctx->dnnl_context.getEngine();
            auto& stm = ctx->dnnl_context.getEngineStream();

            static auto& opsToNames = BertLayer::OpsToNames();

            auto getSubmem = [&](const dnnl::memory::desc& md, const dnnl::memory& buffer, size_t offset) -> dnnl::memory {
                return {
                    md,
                    eng,
                    static_cast<char*>(buffer.get_data_handle()) + offset
                };
            };

            auto query = getSubmem(queryMD, qkvResultBuffer, offsetQuery);
            auto   key = getSubmem(  keyMD, qkvResultBuffer, offsetKey);
            auto value = getSubmem(valueMD, qkvResultBuffer, offsetValue);

            if (joinQKV()) {
                auto qkv_SrcData = qkvIPDesc.ScaledData(inputBufferMem);
                // Query+key+Value
                auto qkv = ctx->PopBuffer(qkvIPDesc.dst_desc());
                ctx->profiler.Profile(opsToNames.at(Ops::query), [&](){
                    qkvIPDesc.Compute(stm, qkv_SrcData, qkv);
                });

                ctx->profiler.Profile(opsToNames.at(Ops::key), [&](){
                    qkvReorder.execute(stm, {
                        {DNNL_ARG_SRC, qkv},
                        {DNNL_ARG_DST, qkvResultBuffer}
                    });
                    stm.wait();
                });
            } else /*!joinQKV*/ {
                auto qkv_SrcData = queryIPDesc.ScaledCachedData(inputBufferMem);
                // Query
                ctx->profiler.Profile(opsToNames.at(Ops::query), [&](){
                    queryIPDesc.Compute(stm, qkv_SrcData, query);
                });

                // Key
                ctx->profiler.Profile(opsToNames.at(Ops::key), [&](){
                    keyIPDesc.Compute(stm, qkv_SrcData, key);
                });

                // Value
                ctx->profiler.Profile(opsToNames.at(Ops::value), [&](){
                    valueIPDesc.Compute(stm, qkv_SrcData, value);
                });
            }

            return { query, key, value };
        }

        dnnl::memory::desc getResultMD() const { return qkvResultMD; }
        dnnl::memory::desc  getQueryMD() const { return queryMD; }
        dnnl::memory::desc    getKeyMD() const { return keyMD; }
        dnnl::memory::desc  getValueMD() const { return valueMD; }
    private:
        bool joinQKV() const {
           return ctx->batch_ <= 1 && !ctx->use_quantization;
        }

        dnnl::memory joinData(const dnnl::memory::dims& dims, const std::vector<dnnl::memory>& srcs) {
            if (srcs.size() == 0) {
                return {};
            }

            auto& eng = ctx->dnnl_context.getEngine();
            auto& stm = ctx->dnnl_context.getEngineStream();

            const dnnl::memory::desc md{dims, srcs.at(0).get_desc().get_data_type(), dnnl::memory::dims{}};
            std::vector<dnnl::memory::desc> descs(srcs.size(), md);

            dnnl::concat::primitive_desc concat_pd{eng, static_cast<int>(dims.size()-1), descs};
            dnnl::memory result{concat_pd.dst_desc(), eng};

            std::unordered_map<int, dnnl::memory> args{{DNNL_ARG_DST, result}};
            for (int i = 0; i < static_cast<int>(srcs.size()); ++i) {
                args.emplace(DNNL_ARG_MULTIPLE_SRC + i, dnnl_wrappers::ReLayoutMemory(srcs[i], md));
            }

            dnnl::concat{concat_pd}.execute(stm, args);
            stm.wait();

            return result;
        }

        std::shared_ptr<BertContext> ctx;
        // Joined query, key, value weight, bias and MatMul op
        BertLayer::InnerProductInferenceDesc qkvIPDesc;
        dnnl::memory qkvWeight;
        dnnl::memory qkvBias;
        dnnl::reorder qkvReorder;

        // Separate query, key and value
        BertLayer::InnerProductInferenceDesc queryIPDesc;
        BertLayer::InnerProductInferenceDesc keyIPDesc;
        BertLayer::InnerProductInferenceDesc valueIPDesc;

        dnnl::memory::desc qkvResultMD;
        dnnl::memory::desc queryMD;
        dnnl::memory::desc keyMD;
        dnnl::memory::desc valueMD;
        size_t offsetQuery;
        size_t offsetKey;
        size_t offsetValue;
    };

    class BatchMatMul1WithScaleBias {
    public:
        BatchMatMul1WithScaleBias(const std::shared_ptr<BertContext>& ctx, const dnnl::memory::desc& query_md, const dnnl::memory::desc& key_md) {
            using namespace dnnl_wrappers;
            using dim = dnnl::memory::dim;
            const dim batch = ctx->batch_;
            const dim tokenSize = ctx->maxTokenSize;
            const dim heads = ctx->numHeads;
            const dim hiddenSize = ctx->hiddenSize;
            const dim headSize = hiddenSize / heads;

            const auto d_dt = ctx->FloatType();

            // B needs to transpose
            // dnnl::memory::format_tag::cab - is not defined
            // const dnnl::memory::dims dnnl_strides__format_tag__adbc{heads * k * m, k, 1, heads * k};

            src_md =   ConvertIPDataDims(query_md, 2).reshape({batch, tokenSize, heads, headSize}).permute_axes(PERMUTE_ACBD);
            weights_md = ConvertIPDataDims(key_md, 2).reshape({batch, tokenSize, heads, headSize}).permute_axes(PERMUTE_ADBC);
            const dnnl::memory::desc    bias_md{};
            dst_md = dnnl::memory::desc{{batch, heads, tokenSize, tokenSize}, d_dt, dnnl::memory::format_tag::abcd};

            input_dims = {batch, 1, 1, tokenSize};
            const dnnl::memory::desc    mask_md{input_dims, dt::f32, dnnl::memory::format_tag::abcd};

            const float scale = 1.f/std::sqrt(static_cast<float>(headSize));
            batchMatMul = std::make_unique<dnnl_wrappers::MatMul>(
                        ctx->dnnl_context.getEngine(),
                        src_md, weights_md, bias_md, dst_md,
                        dnnl_wrappers::BuildAttrs()
                            .Eltwise(dnnl::algorithm::eltwise_linear, scale)
                            .Binary(dnnl::algorithm::binary_add, mask_md)
                            );
        }

        void Compute(dnnl::stream& stm, const dnnl::memory& query, const dnnl::memory& key, const dnnl::memory& input_mask, dnnl::memory& dst) {
            using namespace dnnl_wrappers;

            auto QData = ImmutableDataSource(ReLayoutMemory(query, src_md));
            auto KData = ImmutableDataSource(ReLayoutMemory(key, weights_md));
            auto MaskData = ImmutableDataSource(ReshapeMemory(input_mask, input_dims));
            std::unordered_map<int, std::reference_wrapper<DataSource>> post_ops_data = {{1, std::ref(MaskData)}};

            batchMatMul->ComputeWithPostOps(stm, QData, KData, post_ops_data, dst);
        }

        const dnnl::memory::desc& dst_desc() const {
            return dst_md;
        }

    private:
        dnnl::memory::desc src_md;
        dnnl::memory::desc weights_md;
        dnnl::memory::desc dst_md;
        dnnl::memory::dims input_dims;
        std::unique_ptr<dnnl_wrappers::MatMul> batchMatMul;
    };

    class BatchMatMul2 {
    public:
        BatchMatMul2(const std::shared_ptr<BertContext>& ctx, const dnnl::memory::desc& value_md, const dnnl::memory::desc& attout_md) {
            using namespace dnnl_wrappers;
            using dim = dnnl::memory::dim;
            const dim batch = ctx->batch_;
            const dim tokenSize = ctx->maxTokenSize;
            const dim heads = ctx->numHeads;
            const dim hiddenSize = ctx->hiddenSize;
            const dim headSize = hiddenSize / heads;

            const auto s_dt = ctx->FloatType();

            src_md = dnnl::memory::desc{{batch, heads, tokenSize, tokenSize}, s_dt, dnnl::memory::format_tag::abcd};
            weights_md = ConvertIPDataDims(value_md, 2).reshape({batch, tokenSize, heads, headSize}).permute_axes(PERMUTE_ACBD);
            dst_md    = ConvertIPDataDims(attout_md, 2).reshape({batch, tokenSize, heads, headSize}).permute_axes(PERMUTE_ACBD);

            batchMatMul = std::make_unique<dnnl_wrappers::MatMul>(
                            ctx->dnnl_context.getEngine(),
                            src_md, weights_md, dnnl::memory::desc{}, dst_md,
                            BuildAttrs());
        }

        void Compute(dnnl::stream& stm, const dnnl::memory& qk_result, const dnnl::memory& value, dnnl::memory& dst) {
            using namespace dnnl_wrappers;

            auto QKData = ImmutableDataSource(qk_result);
            auto VData  = ImmutableDataSource(ReLayoutMemory(value, weights_md));
            auto BData  = ImmutableDataSource();
            auto DMem = ReLayoutMemory(dst, dst_md);

            batchMatMul->Compute(stm, QKData, VData, BData, DMem);
        }

        const dnnl::memory::desc& dst_desc() const {
            return dst_md;
        }

    private:
        dnnl::memory::desc src_md;
        dnnl::memory::desc weights_md;
        dnnl::memory::desc dst_md;
        std::unique_ptr<dnnl_wrappers::MatMul> batchMatMul;
    };

private:
    std::shared_ptr<BertContext> ctx;

    std::unique_ptr<QueryKeyValue> queryKeyValue_;

    std::unique_ptr<BatchMatMul1WithScaleBias> batchMatMul1ScaleBias_;

    std::unique_ptr<dnnl_wrappers::SoftMax> softmax_;

    std::unique_ptr<BatchMatMul2> batchMatMul2_;

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
