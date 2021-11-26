// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_LAYER_H_
#define BERT_LAYER_H_

#include <string.h>
#include <math.h>
#include <omp.h>
#include <iostream>

#include "my_types.h"

#include "dnnl_matmul.h"
#include "bert_context.h"
#include "dnnl_batchmatmul.h"
#include "dnnl_softmax.h"
#include "dnnl_layernorm.h"
#include "dnnl_matmul_quant.h"

#define QUANT_INT8
//#define dynamic_quant

dnnl::stream eng_stream(eng);

class BertLayer
{
public:
    // hiddenSize 768 Hidden layer neurons, number of hidden units
    // intermediateSize 3072 feed-forward/filter size dimension 4*hiddensize 
    BertLayer(BertContext &_ctx, int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072) :
    ctx(_ctx) {
        this->maxTokenSize = maxTokenSize;
        this->hiddenSize = hiddenSize;
        this->intermediateSize = intermediateSize;
    }

    ~BertLayer() {}

    float max_matrix(hpj::Matrix<float> &A) {
        float max = -9.0e9;

        for (int i = 0; i < A.Rows(); ++i) {
            float *presult = A.Row(i);
            for (int j = 0; j < A.Cols(); ++j) {
                if(fabs(presult[j]) > max){
                    max = fabs(presult[j]);
                }
            }
        }
        return max;
    }

    void setWeights(const float *_queryWeight, const float *_queryBias,
                    const float *_keyWeight, const float *_keyBias,
                    const float *_valueWeight, const float *_valueBias,
                    const float *_attentionOutputWeight, const float *_attentionOutputBias,
                    const float *_gamma1, const float *_beta1,
                    const float *_intermediateWeight, const float *_intermediateBias,
                    const float *_outputWeight, const float *_outputBias,
                    const float *_gamma2, const float *_beta2) {
        // Merged weights, dimension is like: 768*(768*3)
        hpj::Matrix<float> tmp;
        
#ifdef SEPARATE_QKV
        qkvWeight.Resize(hiddenSize * 3, hiddenSize);
        copyDataWithRows(qkvWeight, 0, hiddenSize, _queryWeight);
        copyDataWithRows(qkvWeight, hiddenSize, hiddenSize, _keyWeight);
        copyDataWithRows(qkvWeight, hiddenSize*2, hiddenSize, _valueWeight);
#else
        tmp.Resize(hiddenSize, hiddenSize * 3);
        copyWeights(tmp, 0, hiddenSize, _queryWeight);
        copyWeights(tmp, hiddenSize, hiddenSize*2, _keyWeight);
        copyWeights(tmp, hiddenSize*2, hiddenSize*3, _valueWeight);
        copyTransposed(qkvWeight, tmp);
#endif
        // Merged bias
        qkvBias.Resize(hiddenSize * 3);
        memcpy(qkvBias.Data(), _queryBias, sizeof(float) * hiddenSize);
        memcpy(qkvBias.Data() + hiddenSize, _keyBias, sizeof(float) * hiddenSize);
        memcpy(qkvBias.Data() + hiddenSize*2, _valueBias, sizeof(float) * hiddenSize);
       
        // Weights for attention output
        attentionOutputWeight.Resize(hiddenSize, hiddenSize);
        copyWeights(attentionOutputWeight, _attentionOutputWeight);
        attentionOutputBias.Resize(hiddenSize);
        memcpy(attentionOutputBias.Data(), _attentionOutputBias, sizeof(float) * hiddenSize);

        gamma1.Resize(hiddenSize);
        beta1.Resize(hiddenSize);
        memcpy(gamma1.Data(), _gamma1, sizeof(float) * hiddenSize);
        memcpy(beta1.Data(), _beta1, sizeof(float) * hiddenSize);

        // intermediate weight and bias
        intermediateWeight.Resize(hiddenSize, intermediateSize);
        copyWeights(intermediateWeight, _intermediateWeight);
        intermediateBias.Resize(intermediateSize);
        memcpy(intermediateBias.Data(), _intermediateBias, sizeof(float) * intermediateSize);

        // output dense weight and bias
        outputWeight.Resize(intermediateSize, hiddenSize);
        copyWeights(outputWeight, _outputWeight);
        outputBias.Resize(hiddenSize);
        memcpy(outputBias.Data(), _outputBias, sizeof(float) * hiddenSize);

        gamma2.Resize(hiddenSize);
        beta2.Resize(hiddenSize);
        memcpy(gamma2.Data(), _gamma2, sizeof(float) * hiddenSize);
        memcpy(beta2.Data(), _beta2, sizeof(float) * hiddenSize);

#ifdef QUANT_INT8
        hpj::Matrix<float> qWeight(qkvWeight, 0, hiddenSize, 0, hiddenSize);
        hpj::Matrix<float> kWeight(qkvWeight, hiddenSize, hiddenSize, 0, hiddenSize);
        hpj::Matrix<float> vWeight(qkvWeight, 2*hiddenSize, hiddenSize, 0, hiddenSize);

        float max_qWeight = max_matrix(qWeight);
        float max_kWeight = max_matrix(kWeight);
        float max_vWeight = max_matrix(vWeight);

        q_WScale = 127/max_qWeight;
        k_WScale = 127/max_kWeight;
        v_WScale = 127/max_vWeight;

        float max_attentionoutWeight = max_matrix(attentionOutputWeight);
        attentionoutWScale = 127/max_attentionoutWeight;

        float max_intermediateWeight = max_matrix(intermediateWeight);
        intermediateWScale = 127/max_intermediateWeight;

        // outputWeight
        float max_outputWeight = max_matrix(outputWeight);
        outWScale = 127/max_outputWeight;

    #ifndef dynamic_quant
        float qkv_src_max = 35.f;

        qkv_SrcScale = 127/qkv_src_max;

        float attentionout_src_max = 7.f;
        float intermediate_src_max = 100.f;
        float out_src_max = 150.f;

        attentionout_SrcScale = 127/attentionout_src_max;
        intermediate_SrcScale = 127/intermediate_src_max;
        out_SrcScale = 127/out_src_max;
    #endif
#endif
    }

    // Do the forward computing for the whole BERT layer
    // input: maxTokenSize x hidden_size
    // actualTokens: #tokens = maxTokenSize - padded_tokens
    hpj::Matrix<float> &forward(hpj::Matrix<float> &inputBuffer) {

#ifdef QUANT_INT8
        sgemm_with_bias_qkv_quant(inputBuffer, qkvWeight, ctx.qkvMatMul, qkvBias);
#else
        sgemm_with_bias_qkv(inputBuffer, qkvWeight, ctx.qkvMatMul, qkvBias);
#endif
        
        hpj::Matrix<float> query(ctx.qkvMatMul, 0, ctx.maxTokenSize, 0, hiddenSize);
        hpj::Matrix<float> key(ctx.qkvMatMul, ctx.maxTokenSize, ctx.maxTokenSize, 0, hiddenSize);
        hpj::Matrix<float> value(ctx.qkvMatMul, 2*ctx.maxTokenSize, ctx.maxTokenSize, 0, hiddenSize);

        batchMatMul_dnnl_1_with_scale_bias(query, key, ctx.qk_result);
        computeSoftmax_only_dnnl();

        batchMatMul_dnnl_2(ctx.qk_result, value, ctx.resultBuffer1);

#ifdef QUANT_INT8

    #ifdef dynamic_quant
        float attentionout_src_max = max_matrix(ctx.resultBuffer1);
        attentionout_SrcScale = 127/attentionout_src_max;
    #endif
        denseWithSum_withSum_quant(ctx.resultBuffer1, attentionOutputWeight, attentionOutputBias, inputBuffer, 
            attentionout_SrcScale, attentionoutWScale);

#else
        denseWithSum_withSum(ctx.resultBuffer1, attentionOutputWeight, attentionOutputBias, inputBuffer);
#endif
        batchnorm_dnnl(inputBuffer, gamma1, beta1);

#ifdef QUANT_INT8

    #ifdef dynamic_quant
        float intermediate_src_max = max_matrix(inputBuffer);
        intermediate_SrcScale = 127/intermediate_src_max;
    #endif
        intermediate_with_erf_quant(inputBuffer, ctx.intermediateBuffer);
#else
        intermediate_with_erf_dst_bf16(inputBuffer, ctx.intermediateBuffer_bf16);
#endif
            
#ifdef QUANT_INT8

    // #ifdef dynamic_quant
    // TODO(rfsaliev) analyze accuracy effect of the dynamic quantization here
    //                improve max_matrix() performance if dyn quant is unavoidable
    #if 1
        float out_src_max = max_matrix(ctx.intermediateBuffer);
        out_SrcScale = 127/out_src_max;
    #endif
        denseWithSum_withSum_quant(ctx.intermediateBuffer, outputWeight, outputBias, inputBuffer, out_SrcScale, outWScale);
#else
        denseWithSum_withSum_src_bf16(ctx.intermediateBuffer_bf16, outputWeight, outputBias, inputBuffer);
#endif
        batchnorm_dnnl(inputBuffer, gamma2, beta2);

        return inputBuffer;
    }

private:
    void copyWeights(hpj::Matrix<float> &w, int start_col, int end_col, const float *data) {
        hpj::Matrix<float> subW(w, 0, w.Rows(), start_col, end_col - start_col);
        copyWeights(subW, data);
    }

    void copyWeights(hpj::Matrix<float> &w, const float *data) {
        for (int i = 0; i < w.Rows(); ++i) {
            for (int j = 0; j < w.Cols(); ++j) {
                w(i, j) = *data++;
            }
        }
    }

    void copyDataWithRows(hpj::Matrix<float> &w, int start_row, int rows, const float *data) {
        int end_row = start_row + rows;
        for (int i = start_row; i < end_row; ++i) {
            for (int j = 0; j < w.Cols(); ++j) {
                w(i, j) = *data++;
            }
        }
    }

    void copyTransposed(hpj::Matrix<float> &dst, hpj::Matrix<float> &src) {
        dst.Resize(src.Cols(), src.Rows());
        for (int i = 0; i < dst.Rows(); ++i) {
            for (int j = 0; j < dst.Cols(); ++j) {
                dst(i, j) = src(j, i);
            }
        }
    }

    void sgemm_with_bias_qkv(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = false;
        // A(128, 768), B(3*768, 768), C(3*128, 768) 
        int m = A.Rows();  // 128
        int k = A.Cols();  // 768
        int n = B.Cols();  // 768
        float *pA = A.Data();

        float *pB1 = B.Data();
        float *pB2 = B.Data() + k*n;
        float *pB3 = B.Data() + 2*k*n;

        float *pC1 = C.Data();
        float *pC2 = C.Data() + m*n;
        float *pC3 = C.Data() + 2*m*n;

        float *pBias1 = bias.Data();
        float *pBias2 = bias.Data() + n;
        float *pBias3 = bias.Data() + 2*n;

        MatMul_with_bias(eng, eng_stream, pA, pB1, pBias1, pC1, m, n, k, wTrans);
        MatMul_with_bias(eng, eng_stream, pA, pB2, pBias2, pC2, m, n, k, wTrans);
        MatMul_with_bias(eng, eng_stream, pA, pB3, pBias3, pC3, m, n, k, wTrans);
    }

    void sgemm_with_bias_qkv_quant(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = false;
        // A(128, 768), B(3*768, 768), C(3*128, 768) 
        int m = A.Rows();  // 128
        int k = A.Cols();  // 768
        int n = B.Cols();  // 768
        float *pA = A.Data();

        float *pB1 = B.Data();
        float *pB2 = B.Data() + k*n;
        float *pB3 = B.Data() + 2*k*n;

        float *pC1 = C.Data();
        float *pC2 = C.Data() + m*n;
        float *pC3 = C.Data() + 2*m*n;

        float *pBias1 = bias.Data();
        float *pBias2 = bias.Data() + n;
        float *pBias3 = bias.Data() + 2*n;

#ifdef dynamic_quant
        float qkv_src_max = max_matrix(A);
        qkv_SrcScale = 127/qkv_src_max;

        MatMul_with_bias_quant(eng, eng_stream, pA, pB1, pBias1, pC1, m, n, k, wTrans, qkv_SrcScale, q_WScale);
        MatMul_with_bias_quant(eng, eng_stream, pA, pB2, pBias2, pC2, m, n, k, wTrans, qkv_SrcScale, k_WScale);
        MatMul_with_bias_quant(eng, eng_stream, pA, pB3, pBias3, pC3, m, n, k, wTrans, qkv_SrcScale, v_WScale);
#else
        MatMul_with_bias_quant(eng, eng_stream, pA, pB1, pBias1, pC1, m, n, k, wTrans, qkv_SrcScale, q_WScale);
        MatMul_with_bias_quant(eng, eng_stream, pA, pB2, pBias2, pC2, m, n, k, wTrans, qkv_SrcScale, k_WScale);
        MatMul_with_bias_quant(eng, eng_stream, pA, pB3, pBias3, pC3, m, n, k, wTrans, qkv_SrcScale, v_WScale);
#endif
    }

    void sgemm_with_sum(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        MatMul_with_sum(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
    }

    void sgemm_with_sum_quant(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias,
                float src_scale, float weight_scale) {
        bool wTrans = (A.Cols() != B.Rows());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        MatMul_with_sum_quant(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans, src_scale, weight_scale);
    }    

    void sgemm_with_sum_src_bf16(hpj::Matrix<bfloat16> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        bfloat16 *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        MatMul_with_sum_src_bf16(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
    }

    void sgemm_with_erf(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        const float factor = sqrt(0.5f);

        MatMul_with_erf(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
    }

    void sgemm_with_erf_quant(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        const float factor = sqrt(0.5f);

        MatMul_with_erf_quant(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans, intermediate_SrcScale, intermediateWScale);
    }

    void sgemm_with_erf_dst_bf16(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<bfloat16> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        bfloat16 *pC = C.Data();
        float *pBias = bias.Data();

        const float factor = sqrt(0.5f);

        MatMul_with_erf_dst_bf16(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
    }

    void denseWithSum_withSum(hpj::Matrix<float> &x, hpj::Matrix<float> &weight, 
                               hpj::Vector<float> &bias, hpj::Matrix<float> &result) {

        sgemm_with_sum(x, weight, result, bias);
    }

    void denseWithSum_withSum_quant(hpj::Matrix<float> &x, hpj::Matrix<float> &weight, 
                               hpj::Vector<float> &bias, hpj::Matrix<float> &result, 
                               float src_scale, float weight_scale) {

        sgemm_with_sum_quant(x, weight, result, bias, src_scale, weight_scale);
    }    

    void denseWithSum_withSum_src_bf16(hpj::Matrix<bfloat16> &x, hpj::Matrix<float> &weight, 
                               hpj::Vector<float> &bias, hpj::Matrix<float> &result) {

        sgemm_with_sum_src_bf16(x, weight, result, bias);
    }

    void batchnorm_dnnl(hpj::Matrix<float> &x, hpj::Vector<float> &gamma, hpj::Vector<float> &beta) {
        assert(x.Rows() == maxTokenSize);
        assert(x.Cols() == hiddenSize);

        int m = x.Rows();
        int n = x.Cols();

        float *pA = x.Data();
        float *pGamma = gamma.Data();
        float *pBeta = beta.Data();

        LayerNorm_with_gamma_beta(eng, eng_stream, pA, pGamma, pBeta, m, n);
    }

    void intermediate_with_erf(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {

        sgemm_with_erf(input, intermediateWeight, output, intermediateBias);
    }

    void intermediate_with_erf_quant(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {

        sgemm_with_erf_quant(input, intermediateWeight, output, intermediateBias);
    }

    void intermediate_with_erf_dst_bf16(hpj::Matrix<float> &input, hpj::Matrix<bfloat16> &output) {

        sgemm_with_erf_dst_bf16(input, intermediateWeight, output, intermediateBias);
    }

    void batchMatMul_dnnl_1_with_scale_bias(hpj::Matrix<float> &A, hpj::Matrix<float> &B, float *c_array[12]){
        bool wTrans = true;
        int m = A.Rows();  // maxTokenSize = 128
        int k = 64;        // 12 * 64 = 768
        int n = B.Rows(); // B needs to transpose

        int batch = 12;
        int lda = hiddenSize;
        int ldb = hiddenSize;
        int ldc = maxTokenSize;

        int batch_stride_a = k;
        int batch_stride_b = k;
        int batch_stride_c = maxTokenSize*maxTokenSize;
        int batch_stride_bias = 0;

        int batch_src = batch;
        int batch_weights = batch;
        int batch_dst = batch;
        int batch_bias = 1;

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = c_array[0];
        
        float *pBias = ctx.magic_value;

        float scale = 0.125f;

        BatchMatMul_with_stride_bias(eng, eng_stream, pA, pB, pC, pBias, m, n, k, lda, ldb, ldc, 
                            batch_stride_a, batch_stride_b, batch_stride_c, batch_stride_bias, 
                            batch_src, batch_weights, batch_dst, batch_bias, scale, wTrans);
    }

    void batchMatMul_dnnl_2(float *a_array[12], hpj::Matrix<float> &B, hpj::Matrix<float> &C){
        bool wTrans = false;
        int m = maxTokenSize;
        int k = maxTokenSize;
        int n = 64;

        float *pA = a_array[0];
        float *pB = B.Data();
        float *pC = C.Data();

        int lda = maxTokenSize;
        int ldb = hiddenSize;
        int ldc = hiddenSize;

        int batch = 12;
        int batch_stride_a = maxTokenSize*maxTokenSize;
        int batch_stride_b = n;
        int batch_stride_c = n;

        BatchMatMul_with_stride(eng, eng_stream, pA, pB, pC, m, n, k, lda, ldb, ldc, 
                        batch_stride_a, batch_stride_b, batch_stride_c, wTrans, batch);
    }

    void computeSoftmax_only_dnnl() {
        int rows = 12*maxTokenSize;
        int cols = maxTokenSize;
        float *pA = ctx.qk_result[0];

        Softmax(eng, eng_stream, pA, rows, cols);
    }

private:
    BertContext &ctx;
    int maxTokenSize;
    int hiddenSize;
    int intermediateSize;

    // Merged query, key, value weighs
    hpj::Matrix<float> qkvWeight;
    // Merged query, key, value bias
    hpj::Vector<float> qkvBias;

    hpj::Matrix<float> attentionOutputWeight;
    hpj::Vector<float> attentionOutputBias;

    hpj::Vector<float> gamma1, beta1;
    hpj::Vector<float> gamma2, beta2;

    hpj::Matrix<float> intermediateWeight;
    hpj::Vector<float> intermediateBias;

    hpj::Matrix<float> outputWeight;
    hpj::Vector<float> outputBias;

    float qkv_SrcScale;

    float q_WScale;
    float k_WScale;
    float v_WScale;

    float attentionout_SrcScale;
    float intermediate_SrcScale;
    float out_SrcScale;

    float attentionoutWScale;
    float intermediateWScale;
    float outWScale;
};

#endif
