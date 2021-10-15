// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_CONTEXT_H_
#define BERT_CONTEXT_H_

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
        //intermediateBuffer_bf16.Resize(maxTokenSize, intermediateSize);

        qk_resultBuffer.Resize(12*maxTokenSize, maxTokenSize);
        //qk_resultBuffer.Resize(maxTokenSize, maxTokenSize*12);

        
        for (int i = 0; i < 12; ++i) {
            //qk_result[i] = (float *)aligned_alloc(64, sizeof(float) * maxTokenSize * maxTokenSize);
            qk_result[i] = qk_resultBuffer.Data() + i * maxTokenSize * maxTokenSize;
            exp_buffer[i] = (float *)aligned_alloc(64, sizeof(float) * maxTokenSize);
        }

        magic_value = (float *)aligned_alloc(64, sizeof(float) * maxTokenSize);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) { num_threads = omp_get_num_threads(); }
        }

#ifndef __INTEL_COMPILER 
        erf_buffer = new float * [num_threads];
        for (int i = 0; i < num_threads; ++i) {
            erf_buffer[i] = (float *)aligned_alloc(64, sizeof(float) * intermediateSize);
        }
#endif
    }

    virtual ~BertContext() {
        for (int i = 0; i < 12; ++i) {
            //free(qk_result[i]);
            free(exp_buffer[i]);
            //qk_result[i] = NULL;
            exp_buffer[i] = NULL;
        }
        free(magic_value);
        magic_value = NULL;

#ifndef __INTEL_COMPILER
        for (int i = 0; i < num_threads; ++i) {
            free(erf_buffer[i]);
        }
        delete[] erf_buffer;
        erf_buffer = NULL;
#endif
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
    //hpj::Matrix<bfloat16> intermediateBuffer_bf16;
    // Store the BatchMatMul result of query and key
    float *qk_result[12];
    // Store the result of exp for each line
    float *exp_buffer[12];

    hpj::Matrix<float> qk_resultBuffer;

    // Magic value: 0 or -10000
    float *magic_value;

    int num_threads;
#ifndef __INTEL_COMPILER 
    float **erf_buffer;
#endif
};

#endif