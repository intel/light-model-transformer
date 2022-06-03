// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_CONTEXT_H_
#define BERT_CONTEXT_H_

#include "dnnl_common.h"

#include <cstdlib>
#include <omp.h>
#include <vector>
#include <memory>
#include <cassert>

#define QUANT_INT8

// TODO: (kchutkie) refactor these into runtime arguments.
// Maybe pack all ctor arguments into a struct?
template <class InputT = float, class BatchInputT = float,
    class = std::enable_if_t<
        std::is_arithmetic<InputT>::value &&
        std::is_arithmetic<BatchInputT>::value>
>
class BertContext {

    using dt = dnnl::memory::data_type;
    using dims = dnnl::memory::dims;
    using dim = dnnl::memory::dim;

public:
    using input_t = InputT;
    using batch_input_t = BatchInputT;

    // All BERT models use same head size - 64
    // * Base: hiddenSize = 768, heads = 12
    // * Large: hiddenSize = 1024, heads = 16
    // TODO(rfsaliev) review correlation with the 'NumAttentionHeads' attribute
    static constexpr int head_size = 64;
    static constexpr int tensors_per_layer = 16;

    BertContext(int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072, int batch = 1, int numLayers = 12)
        : maxTokenSize{maxTokenSize}
        , hiddenSize{hiddenSize}
        , intermediateSize{intermediateSize}
        , batch_{batch}
        , numLayers{numLayers}
        , numHeads{hiddenSize / head_size}
        , query{dnnl::memory::desc{{batch * maxTokenSize, hiddenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , key  {dnnl::memory::desc{{batch * maxTokenSize, hiddenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , value{dnnl::memory::desc{{batch * maxTokenSize, hiddenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , resultBuffer1{dnnl::memory::desc{{batch * maxTokenSize, hiddenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , intermediateBuffer{dnnl::memory::desc{{batch * maxTokenSize, intermediateSize}, DnnlDataType<input_t>::value, dims{}}, dnnl_context.getEngine()}
        , qk_resultBuffer{dnnl::memory::desc{{batch, hiddenSize / head_size, maxTokenSize, maxTokenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , input_mask{dnnl::memory::desc{{batch, 1, 1, maxTokenSize}, dt::f32, dims{}}, dnnl_context.getEngine()} // Broadcast dimensions M and N
    {
        assert(hiddenSize % head_size == 0);
    }

    /**
     * @brief Set the input mask for a specific batch element
     * 
     * @tparam T Type of the raw data pointer
     * @param input_mask Pointer to the raw data containing the input mask
     * @param size Number of elements in the mask buffer
     * @param batch Which element of the batch to set
     */
    template <typename T>
    void setInputMask(const T* mask, dim size, dim batch)
    {
        // TODO(rfsaliev) utilize dnnl::eltwise(linear) instead
        auto desc = input_mask.get_desc();
        auto dims = desc.dims();
        dim m_stride = dims[3];
        dim head_stride = m_stride * dims[2];
        dim batch_stride = head_stride * dims[1];
        assert(m_stride > 0);
        assert(head_stride > 0);
        assert(batch_stride > 0);
        
        if (batch >= dims[0] || batch < 0)
        {
            throw std::out_of_range("Index to copy the mask buffer to is invalid.");
        }
        if (size < batch_stride)
        {
            throw std::invalid_argument("Not enough elements to copy from the mask buffer.");
        }

        MemoryAccessor<float> input_mask_acc(input_mask);
        for (int h = 0; h < dims[1]; ++h)
        {
            for (int m = 0; m < dims[2]; ++m)
            {
                for (int n = 0; n < dims[3]; ++n)
                {
                    input_mask_acc.Data()[batch * batch_stride + h * head_stride + m * m_stride + n] = -10000.f * (1.f - mask[n]);
                }
            }

        }

    }

    int maxTokenSize;
    int hiddenSize;
    int intermediateSize;
    int batch_;
    int numLayers;
    int numHeads;
    DnnlCommon dnnl_context;

    // Store the result of input*qkvWeight
    dnnl::memory query;
    dnnl::memory key;
    dnnl::memory value;
    // Buffer like the dimesion of 128x768
    dnnl::memory resultBuffer1;
    // Buffer to store the result of intermediate
    dnnl::memory intermediateBuffer;
    // Store the BatchMatMul result of query and key
    dnnl::memory qk_resultBuffer;

    // Magic value: 0 or -10000
    dnnl::memory input_mask;
};



// Helper class to build a BertContext.
// In TF op we get some of the context arguments
// later than others, so we need a way to store them. 
// Also makes the arguments of the BertContext ctor more
// manageable.
class BertContextBuilder {
public:
    template <class InputT, class BatchInputT>
    std::shared_ptr<BertContext<InputT, BatchInputT>> Build() {
        return std::make_shared<BertContext<InputT, BatchInputT>>(max_token_size, hidden_size, intermediate_size,
                                                                  batch_size, num_layers);
    }

    void MaxTokenSize(int max_token_size) { this->max_token_size = max_token_size; }
    void HiddenSize(int hidden_size) { this->hidden_size = hidden_size; }
    void IntermediateSize(int intermediate_size) { this->intermediate_size = intermediate_size; }
    void BatchSize(int batch_size) { this->batch_size = batch_size; }
    void NumLayers(int num_layers) { this->num_layers = num_layers; }
    void NumAttentionHeads (int num_attention_heads) { this->num_attention_heads = num_attention_heads; }

    int MaxTokenSize() const { return this->max_token_size; }
    int HiddenSize() const { return this->hidden_size; }
    int IntermediateSize() const { return this->intermediate_size; }
    int BatchSize() const { return this->batch_size; }
    int NumLayers() const { return this->num_layers; }
    int NumAttentionHeads() const { return this->num_attention_heads; }

private:
    int max_token_size = 128;
    int hidden_size = 768;
    int intermediate_size = 3072;
    int batch_size = 1;
    int num_layers = 12;
    int num_attention_heads = 12;
};

#endif
