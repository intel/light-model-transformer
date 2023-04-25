// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bert_op.hpp"

#include "bert_layer_quant_int8.h"
#include "bert_context.h"

#include <oneapi/dnnl/dnnl.hpp>

#include <vector>
#include <algorithm>

namespace bert_op
{

namespace
{
class TensorAdapter
{
public:

    static dnnl::memory AsDnnlMemory(const torch::Tensor& tensor, dnnl::engine& engine, bool copy = false)
    {
        auto data_type = AsDnnlDataType(tensor.scalar_type());
        auto tensor_dims = TensorDims(tensor);
        auto tensor_strides = TensorStrides(tensor);
        dnnl::memory::desc md{tensor_dims, data_type, tensor_strides};
        return AsDnnlMemory(tensor, md, engine);
    }

    static dnnl::memory AsDnnlMemory(const torch::Tensor& tensor, const dnnl::memory::desc& md, dnnl::engine& engine, bool copy = false)
    {
        return dnnl::memory{md, engine, tensor.data_ptr()};
    }

private:

    static dnnl::memory::dims TensorDims(const torch::Tensor& t)
    {
        dnnl::memory::dims dims;
        auto sizes = t.sizes();
        std::copy(std::begin(sizes), std::end(sizes), std::back_inserter(dims));
        return dims;
    }

    static dnnl::memory::dims TensorStrides(const torch::Tensor& t)
    {
        dnnl::memory::dims dims;
        auto strides = t.strides();
        std::copy(std::begin(strides), std::end(strides), std::back_inserter(dims));
        return dims;
    }

    static dnnl::memory::data_type AsDnnlDataType(torch::ScalarType t)
    {
        using dt = dnnl::memory::data_type;
        switch(t) {
            case torch::kFloat32:
                return dt::f32;
            case torch::kInt8:
                return dt::s8;
            case torch::kInt32:
                return dt::s32;
            case torch::kBFloat16:
                return dt::bf16;
            default:
                throw std::invalid_argument("Unsupported torch::ScalarType.");
        }
    }

};
}

void BertOp::Configure(int64_t max_seq_len = 128, int64_t hidden_size = 768, int64_t intermediate_size = 3072, int64_t batch_size = 1,
                int64_t num_layers = 12, bool use_quantization = false, bool use_bfloat16 = false,
                bool calibrate_quant_factors = false)
{
    context_ = std::make_shared<BertContext>(max_seq_len, hidden_size, intermediate_size, batch_size, num_layers,
                                                  use_quantization, use_bfloat16, calibrate_quant_factors);

    layers_.reserve(num_layers);
    for(int i = 0; i < num_layers; ++i)
    {
        layers_.emplace_back(std::make_unique<BertLayer>(context_));
    }
}

std::vector<double> BertOp::GetQuantizationFactors() const
{
    std::vector<double> factors;
    for(const auto& layer: layers_)
    {
        auto layer_factors = layer->QuantFactors().AsVector();
        std::copy(std::begin(layer_factors), std::end(layer_factors), std::back_inserter(factors));
    }

    return factors;
}


void BertOp::Initialize(const std::vector<torch::Tensor>& parameters, const std::vector<double>& quant_factors)
{
    StoreParameters(parameters);

    auto get_param = [this](size_t layer, size_t offset)
    {
        return parameters_.at(BertContext::tensors_per_layer * layer + offset);
    };

    if (context_->use_quantization && quant_factors.size() != QuantizationFactors::num_parameters * layers_.size())
    {
        throw std::invalid_argument("Quantization was enabled, but the quant_factors vector does not contain the "
                                    "correct number of values.");
    }

    for(size_t i = 0; i < layers_.size(); ++i)
    {
        auto& layer = layers_.at(i);

        auto query_weight = get_param(i, 0);
        auto query_bias = get_param(i, 1);
        auto key_weight = get_param(i, 2);
        auto key_bias = get_param(i, 3);
        auto value_weight = get_param(i, 4);
        auto value_bias = get_param(i, 5);

        auto attention_weight = get_param(i, 6);
        auto attention_bias = get_param(i, 7);

        auto attention_norm_weight = get_param(i, 8);
        auto attention_norm_bias = get_param(i, 9);

        auto intermediate_weight = get_param(i, 10);
        auto intermediate_bias = get_param(i, 11);

        auto output_weight = get_param(i, 12);
        auto output_bias = get_param(i, 13);

        auto output_norm_weight = get_param(i, 14);
        auto output_norm_bias = get_param(i, 15);

        QuantizationFactors factors;
        if (context_->use_quantization)
        {
            // Take the slice of quantzation factor values and convert to float.
            std::vector<float> factor_values(std::begin(quant_factors) + i * QuantizationFactors::num_parameters,
                                            std::begin(quant_factors) + (i + 1) * QuantizationFactors::num_parameters);
            factors = QuantizationFactors::FromVector(factor_values);
        }

        layer->setWeights(query_weight, query_bias,
                          key_weight, key_bias,
                          value_weight, value_bias,
                          attention_weight, attention_bias,
                          attention_norm_weight, attention_norm_bias,
                          intermediate_weight, intermediate_bias,
                          output_weight, output_bias,
                          output_norm_weight, output_norm_bias,
                          factors);
    }
}

void BertOp::StoreParameters(const std::vector<torch::Tensor>& parameters)
{
    auto clone_memory = [this](const dnnl::memory& m, const dnnl::memory::desc& md = {}) -> dnnl::memory
    {
        auto target_md = md.is_zero() ? m.get_desc() : md;
        auto target = dnnl::memory{target_md, context_->dnnl_context.getEngine()};
        auto r = dnnl::reorder{m, target};
        r.execute(context_->dnnl_context.getEngineStream(),
        {
            {DNNL_ARG_FROM, m},
            {DNNL_ARG_TO, target}
        });
        context_->dnnl_context.getEngineStream().wait();
        return target;
    };

    parameters_.reserve(parameters.size());
    for(auto& parameter: parameters)
    {
        auto m = TensorAdapter::AsDnnlMemory(parameter, context_->dnnl_context.getEngine());
        if (parameter.sizes().size() == 1)
        {
            parameters_.emplace_back(clone_memory(m));
        }
        else if (parameter.sizes().size() == 2)
        {
            // This clone_memory call causes a reorder that physically transposes the 2D parameters.
            // The original BertEncoder uses torch.nn.Linear, which performs `y = x*A^T + b`, so we transpose the `A^T`
            // back to `A` for the BertLayers to consume.
            auto md = dnnl::memory::desc{m.get_desc().get_dims(), m.get_desc().get_data_type(), dnnl::memory::format_tag::ba};
            parameters_.emplace_back(clone_memory(m, md));
        }
        else
        {
            throw std::invalid_argument("BertEncoder parameters must have either 1 or 2 dimensions.");
        }
    }

}


torch::Tensor BertOp::Forward(torch::Tensor embeddings, torch::Tensor attention_mask)
{
    auto embedding_input = TensorAdapter::AsDnnlMemory(embeddings, context_->dnnl_context.getEngine());
    auto input_mask = TensorAdapter::AsDnnlMemory(attention_mask, context_->dnnl_context.getEngine());
    auto input = this->layers_.front()->PrepareInput(embedding_input);

    for (const auto& bert_layer : this->layers_)
    {
        bert_layer->forward(input, input_mask);
    }

    this->layers_.back()->ProcessResult(input, embedding_input);
    return embeddings;
}


TORCH_LIBRARY(bert_op, m) {
    m.class_<BertOp>("BertOp")
    .def(torch::init())
    .def("configure", &BertOp::Configure)
    .def("get_quantization_factors", &BertOp::GetQuantizationFactors)
    .def("initialize", &BertOp::Initialize)
    .def("forward", &BertOp::Forward)
    ;
}

} // namespace bert_op
