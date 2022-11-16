// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef LIBRARIES_AI_PERFORMANCE_MODELS_BERT_TENSOR_VALIDATOR_H
#define LIBRARIES_AI_PERFORMANCE_MODELS_BERT_TENSOR_VALIDATOR_H

#include "dnnl.hpp"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

dnnl::memory::data_type AsDnnlDataType(tensorflow::DataType data_type)
{
    using dt = dnnl::memory::data_type;
    using namespace tensorflow;
    switch(data_type) {
        case DT_FLOAT:
            return dt::f32;
        case DT_INT8:
            return dt::s8;
        case DT_INT32:
            return dt::s32;
        case DT_BFLOAT16:
            return dt::bf16;
        default:
            throw std::invalid_argument("Unsupported tensorflow::DataType");
    }
}

class TensorAdapter
{
using dims = dnnl::memory::dims;
using dim = dnnl::memory::dim;
using dt = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;
public:
    enum class TensorType {
    Embedded,
    Mask,
    QkvWeight,
    QkvBias,
    AttentionWeight,
    AttentionBias,
    NormGamma,
    NormBeta,
    IntermediateWeight,
    IntermediateBias,
    OutputWeight,
    OutputBias
    };

    TensorAdapter() = default;
    TensorAdapter(dim batch, dim max_token_size, dim hidden_size, dim num_attention_heads, dim intermediate_size)
    {
        Init(batch, max_token_size, hidden_size, num_attention_heads, intermediate_size);
    }

    void Init(dim batch, dim max_token_size, dim hidden_size, dim num_attention_heads, dim intermediate_size)
    {
        dim head_size = hidden_size / num_attention_heads;

        // In different TF1 and TF2 models the dimensions of the same tensor may differ,
        // though the memory layout and total size remain consistent.

        allowed_dims.emplace(TensorType::Embedded, dims{batch * max_token_size, hidden_size});
        allowed_dims.emplace(TensorType::Embedded, dims{batch, max_token_size, hidden_size});

        // Input mask is an exception to the above, with TF-Hub models duplicating the mask over max_token_size rows.
        allowed_dims.emplace(TensorType::Mask, dims{batch, 1, max_token_size});
        allowed_dims.emplace(TensorType::Mask, dims{batch, max_token_size, max_token_size});
        allowed_dims.emplace(TensorType::Mask, dims{batch, 1, 1, max_token_size});

        allowed_dims.emplace(TensorType::QkvWeight, dims{hidden_size, hidden_size});
        allowed_dims.emplace(TensorType::QkvWeight, dims{hidden_size, num_attention_heads, head_size});

        allowed_dims.emplace(TensorType::QkvBias, dims{hidden_size});
        allowed_dims.emplace(TensorType::QkvBias, dims{num_attention_heads, head_size});

        allowed_dims.emplace(TensorType::AttentionWeight, dims{hidden_size, hidden_size});
        allowed_dims.emplace(TensorType::AttentionWeight, dims{num_attention_heads, head_size, hidden_size});

        allowed_dims.emplace(TensorType::AttentionBias, dims{hidden_size});

        allowed_dims.emplace(TensorType::NormGamma, dims{hidden_size});

        allowed_dims.emplace(TensorType::NormBeta, dims{hidden_size});

        allowed_dims.emplace(TensorType::IntermediateWeight, dims{hidden_size, intermediate_size});

        allowed_dims.emplace(TensorType::IntermediateBias, dims{intermediate_size});

        allowed_dims.emplace(TensorType::OutputWeight, dims{intermediate_size, hidden_size});

        allowed_dims.emplace(TensorType::OutputBias, dims{hidden_size});
    }


    /**
     * @brief Return the data pointer of the tensor as float*, if the tensor dimensions are valid.
     * 
     * @param tensor The tensor.
     * @param tensor_type TensorType of the tensor, used to fetch the list of allowed dimensions.
     * @return The tensor data.
     * @throws std::runtime_error if dimensions are invalid.
     */
    template<typename T, std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
    T* GetValidatedTensor(const tensorflow::Tensor& tensor, TensorType tensor_type) const
    {
        ThrowIfDimsInvalid(tensor, tensor_type);
        return reinterpret_cast<T*>(const_cast<char*>(tensor.tensor_data().data()));
    }

    dnnl::memory GetValidatedTensor(const tensorflow::Tensor& tensor, TensorType tensor_type, dnnl::engine& engine) const
    {
        ThrowIfDimsInvalid(tensor, tensor_type); // This can probably be removed, we can rely on dnnl::memory::desc::reshape() in BertLayer
        return AsDnnlMemory(tensor, engine);
    }

    dnnl::memory AsDnnlMemory(const tensorflow::Tensor& tensor, dnnl::engine& engine) const
    {
        auto data_type = AsDnnlDataType(tensor.dtype());
        auto tensor_dims = TensorDims(tensor);
        dnnl::memory::desc md{tensor_dims, data_type, dims{}};
        return AsDnnlMemory(tensor, md, engine);
    }

    dnnl::memory AsDnnlMemory(const tensorflow::Tensor& tensor, dnnl::memory::desc md, dnnl::engine& engine) const
    {
        auto data = reinterpret_cast<void*>(const_cast<char*>(tensor.tensor_data().data()));
        return dnnl::memory{md, engine, data};
    }

    void ThrowIfDimsInvalid(const tensorflow::Tensor& tensor, TensorType tensor_type) const
    {
        // Fetch the dimensions allowed for this TensorType
        auto range = allowed_dims.equal_range(tensor_type);
        if (range.first == end(allowed_dims))
        {
            throw std::runtime_error("Invalid TensorType. This can mean that an invalid integer value was cast to "
                                     "TensorType, or that the TensorAdapter must be updated to support a new value.");
        }

        auto first = range.first;
        auto last = range.second;

        auto tensor_dims = TensorDims(tensor);
        if (std::any_of(first, last, [&tensor_dims](auto it) {
            return it.second == tensor_dims;
        }))
        {
            return;
        }

        try
        {
            // operator<< can technically throw exceptions,
            // hence the 'try' block with a default runtime error message as fallback.
            std::stringstream ss;
            ss << "Invalid tensor dimensions. Dimensions are: ( ";
            std::copy(begin(tensor_dims), end(tensor_dims), std::ostream_iterator<dim>(ss, " "));
            ss << "). Allowed dimensions are: ";
            for (auto it = first; it != last; ++it)
            {
                ss << "( ";
                std::copy(begin(it->second), end(it->second), std::ostream_iterator<dim>(ss, " "));
                ss << ") ";
            }
            throw std::runtime_error(ss.str());
        }
        catch(const std::ios_base::failure& e)
        {
            // Continue to throw a fallback runtime error with a generic message.
        }
        throw std::runtime_error("Invalid tensor dimensions. (Could not print tensor dims.)");
    }

private:

    /**
     * @brief Get tensor dimensions as dnnl::memory::dims.
     * 
     * @param tensor The input tensor.
     * @return Tensor dimensions.
     */
    dims TensorDims(const tensorflow::Tensor& tensor) const
    {
        dims tensor_dims;
        auto tmp_tensor_dims = tensor.shape().dim_sizes();
        std::copy(begin(tmp_tensor_dims), end(tmp_tensor_dims), std::back_inserter(tensor_dims));
        return tensor_dims;
    }

    std::unordered_multimap<TensorType, dims> allowed_dims;
};

#endif
