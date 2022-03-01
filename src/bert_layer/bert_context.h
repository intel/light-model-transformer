// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_CONTEXT_H_
#define BERT_CONTEXT_H_

#include "dnnl_common.h"

#include <cstdlib>
#include <omp.h>
#include <vector>
#include <memory>

#define QUANT_INT8

class BertContext {
    using dt = dnnl::memory::data_type;
    using dims = dnnl::memory::dims;
public:
#ifdef QUANT_INT8
    using input_t = int8_t;
#else
    using input_t = float;
#endif

    // All BERT models use same head size - 64
    // * Base: hiddenSize = 768, heads = 12
    // * Large: hiddenSize = 1024, heads = 16
    // TODO(rfsaliev) review correlation with the 'NumAttentionHeads' attribute
    static constexpr int head_size = 64;

    BertContext(int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072)
        : maxTokenSize{maxTokenSize}
        , hiddenSize{hiddenSize}
        , intermediateSize{intermediateSize}
        , query{dnnl::memory::desc{{maxTokenSize, hiddenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , key  {dnnl::memory::desc{{maxTokenSize, hiddenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , value{dnnl::memory::desc{{maxTokenSize, hiddenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , resultBuffer1{dnnl::memory::desc{{maxTokenSize, hiddenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , intermediateBuffer{dnnl::memory::desc{{maxTokenSize, intermediateSize}, DnnlDataType<input_t>::value, dims{}}, dnnl_context.getEngine()}
        , qk_resultBuffer{dnnl::memory::desc{{hiddenSize / head_size, maxTokenSize, maxTokenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
        , magic_value{dnnl::memory::desc{{1,1,maxTokenSize}, dt::f32, dims{}}, dnnl_context.getEngine()}
    {
        assert(hiddenSize % head_size == 0);
    }

    // Set input mask
    template <typename T>
    void setInputMask(const T *input_mask)
    {
        // TODO(rfsaliev) utilize dnnl::eltwise(linear) instead
        MemoryAccessor<float> magic_value_acc(magic_value);
        assert(magic_value.get_desc().get_size() >= maxTokenSize * sizeof(float));
        for (int i = 0; i < maxTokenSize; ++i)
        {
            magic_value_acc.Data()[i] = -10000.0f * (1 - input_mask[i]);
        }
    }

    int maxTokenSize;
    int hiddenSize;
    int intermediateSize;
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
    dnnl::memory magic_value;
};

#endif
