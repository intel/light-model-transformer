// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_CONTEXT_H_
#define BERT_CONTEXT_H_

#include "my_types.h"

#include <cstdlib>
#include <omp.h>
#include <vector>
#include <memory>


#define SEPARATE_QKV

class BertContext {
public:
    BertContext(int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072) {
        this->maxTokenSize = maxTokenSize;
        this->hiddenSize = hiddenSize;
        this->intermediateSize = intermediateSize;

#ifdef SEPARATE_QKV
        qkvMatMul.Resize(maxTokenSize*3, hiddenSize);
#else
        qkvMatMul.Resize(maxTokenSize, hiddenSize*3);
#endif
        resultBuffer1.Resize(maxTokenSize, hiddenSize);
        resultBuffer2.Resize(maxTokenSize, hiddenSize);
        intermediateBuffer.Resize(maxTokenSize, intermediateSize);

        qk_resultBuffer.Resize(12*maxTokenSize, maxTokenSize);

        qk_result.reserve(12);

        for (int i = 0; i < 12; ++i) {
            qk_result.emplace_back(qk_resultBuffer.Data() + i * maxTokenSize * maxTokenSize);
        }

        magic_value.reset((float *)aligned_alloc(64, sizeof(float) * maxTokenSize));
    }

    virtual ~BertContext() {
    }

    // Set input mask
    template <typename T>
    void setInputMask(const T *input_mask)
    {
        for (int i = 0; i < maxTokenSize; ++i)
        {
            this->magic_value.get()[i] = -10000.0f * (1 - input_mask[i]);
        }
    }

    int maxTokenSize;
    int hiddenSize;
    int intermediateSize;

    // Store the result of input*qkvWeight
    hpj::Matrix<float> qkvMatMul;
    // Buffer like the dimesion of 128x768
    hpj::Matrix<float> resultBuffer1, resultBuffer2;
    // Buffer to store the result of intermediate
    hpj::Matrix<float> intermediateBuffer;
    // Store the BatchMatMul result of query and key
    std::vector<float*> qk_result{};

    hpj::Matrix<float> qk_resultBuffer;

    // Magic value: 0 or -10000
    std::unique_ptr<float, decltype(&free)> magic_value = std::unique_ptr<float, decltype(&free)>(0, &free);
};

#endif