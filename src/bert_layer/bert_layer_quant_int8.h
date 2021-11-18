// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_LAYER_H_
#define BERT_LAYER_H_

#include <new>
#include <map>
#include <numeric>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <iostream>
#include <immintrin.h>
#include "my_types.h"
#include "timer.h"

#include "dnnl.hpp"
#include "dnnl_debug.h"

#if defined(ONEDNN_2_2)
#include "common/bfloat16.hpp"
extern "C" {
dnnl_status_t DNNL_API dnnl_gemm_bf16bf16f32(char transa, char transb,
         dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, float alpha,
         const dnnl::impl::bfloat16_t *A, dnnl_dim_t lda,
         const dnnl::impl::bfloat16_t *B, dnnl_dim_t ldb, float beta,
         float *C, dnnl_dim_t ldc);
}
#elif defined(ONEDNN_1_3)
#include "bfloat16.hpp"
#endif

#include "dnnl_matmul.h"
#include "bert_context.h"
#include "dnnl_batchmatmul.h"
#include "dnnl_softmax.h"
#include "dnnl_layernorm.h"
#include "dnnl_matmul_quant.h"

#define QUANT_INT8
//#define dynamic_quant

dnnl::stream eng_stream(eng);

memory::dim product(const memory::dims &dims) {
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
            std::multiplies<memory::dim>());
}

engine cpu_engine;
stream cpu_stream;
std::map<std::string, inner_product_forward::primitive_desc *> g_net_fwd_prim_desc;
std::map<std::string, primitive *> g_net_fwd_prim;
std::map<std::string, memory *> g_fc_weights_memory;

class BertLayer
{
public:
    // hiddenSize 768 隐层神经元、隐藏单元数
    // intermediateSize 3072 feed-forward/filter size 升维维度 4*hiddensize 
    BertLayer(BertContext &_ctx, int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072) :
    ctx(_ctx) {
        this->maxTokenSize = maxTokenSize;
        this->hiddenSize = hiddenSize;
        this->intermediateSize = intermediateSize;
    }

    virtual ~BertLayer() {

    }

    template <typename T>
    void find_min_max(const std::vector<T> &v, float &min_value, float &max_value) {
        min_value = max_value = v[0];
        for (auto &e : v) {
            min_value = std::min<float>(min_value, e);
            max_value = std::max<float>(max_value, e);
        }
    }

    float max_matrix_wrong(hpj::Matrix<float> &A) {
        //int num_threads = 24; // hard core, may need to change 
        int num_threads = omp_get_max_threads();
        float* max = (float *)malloc(sizeof(float) * num_threads);
        for(int i = 0; i < num_threads; i++){
            max[i] = -9.0e9;
        }
        #pragma omp parallel for
        for (int i = 0; i < A.Rows(); ++i) {
            float *presult = A.Row(i);
            int id = omp_get_thread_num();
            #pragma omp simd
            for (int j = 0; j < A.Cols(); ++j) {
                if(fabs(presult[j]) > max[id]){
                    max[id] = fabs(presult[j]);
                }
            }
        }
        float real_max = max[0];
        for(int i = 1; i < num_threads; i++){
            if(max[i] > real_max){
                real_max = max[i];  
            }  
        }
        free(max);
        //std::cout << "Max value in matrix = " << max << std::endl;
        return real_max;
    }

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

        //printf("max_qWeight = %f, max_kWeight = %f, max_vWeight = %f\n", max_qWeight, max_kWeight, max_vWeight);
        //printf("max_attentionoutWeight = %f, max_intermediateWeight = %f, max_outputWeight = %f\n", max_attentionoutWeight, max_intermediateWeight, max_outputWeight);
    #ifndef dynamic_quant
        float qkv_src_max = 100.f;

	    qkv_SrcScale = 127/qkv_src_max;

        float attentionout_src_max = 100.f;
        float intermediate_src_max = 100.f;
        float out_src_max = 100.f;

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
        // Query, Key, Value computed together
        //printf("qkvWeight, rows = %d, cols = %d\n", qkvWeight.Rows(), qkvWeight.Cols());

#ifdef QUANT_INT8
        sgemm_with_bias_qkv_quant(inputBuffer, qkvWeight, ctx.qkvMatMul, qkvBias);
#else
        sgemm_with_bias_qkv(inputBuffer, qkvWeight, ctx.qkvMatMul, qkvBias);
#endif
        
        hpj::Matrix<float> query(ctx.qkvMatMul, 0, ctx.maxTokenSize, 0, hiddenSize);
        hpj::Matrix<float> key(ctx.qkvMatMul, ctx.maxTokenSize, ctx.maxTokenSize, 0, hiddenSize);
        hpj::Matrix<float> value(ctx.qkvMatMul, 2*ctx.maxTokenSize, ctx.maxTokenSize, 0, hiddenSize);

        batchMatMul_dnnl_1_with_scale_bias(query, key, ctx.qk_result);
        //computeSoftmax_only(actualTokens);
        computeSoftmax_only_dnnl();

        //batchMatMul_2(ctx.qk_result, value, ctx.resultBuffer1);
        batchMatMul_dnnl_2(ctx.qk_result, value, ctx.resultBuffer1);

#ifdef QUANT_INT8

    #ifdef dynamic_quant
        float attentionout_src_max = max_matrix(ctx.resultBuffer1);
        attentionout_SrcScale = 127/attentionout_src_max;
        //printf("attentionout_src_max = %f\n", attentionout_src_max);
    #endif
        denseWithSum_withSum_quant(ctx.resultBuffer1, attentionOutputWeight, attentionOutputBias, inputBuffer, 
            attentionout_SrcScale, attentionoutWScale);

#else
        denseWithSum_withSum(ctx.resultBuffer1, attentionOutputWeight, attentionOutputBias, inputBuffer);
#endif
    
        //batchnorm(inputBuffer, gamma1, beta1);
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

    #ifdef dynamic_quant
        float out_src_max = max_matrix(ctx.intermediateBuffer);
        out_SrcScale = 127/out_src_max;

        /*
        printf("out_src_max = %f\n", out_src_max);
        {
        std::vector<float> tmp_vec(ctx.intermediateBuffer.Data(), ctx.intermediateBuffer.Data()+128*3072);
        float min = 0;
        float max = 0;
        find_min_max(tmp_vec, min, max);
        printf("out_src_max， min = %f, max = %f\n", min, max);
        }
        */
        //print_matrix(ctx.intermediateBuffer);
    #endif
        denseWithSum_withSum_quant(ctx.intermediateBuffer, outputWeight, outputBias, inputBuffer, out_SrcScale, outWScale);
#else
        denseWithSum_withSum_src_bf16(ctx.intermediateBuffer_bf16, outputWeight, outputBias, inputBuffer);
#endif
        //exit(0);
        //batchnorm(inputBuffer, gamma2, beta2);
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

    void dumpMatrix(hpj::Matrix<float> &m) {
        int cols = m.Cols();
        for (int i = 0; i < m.Rows(); ++i) {
            if (m.Cols() < 10) {
                for (int j = 0; j < m.Cols(); ++j) {
                    std::cout << m(i, j) << " ";
                }
            } else {
                std::cout << m(i, 0) << " " << m(i, 1) << " " << m(i, 2) << " ... " << m(i, cols-3) << " " <<  m(i, cols-2) << " " <<  m(i, cols-1);
            }
            std::cout << std::endl;
        }
    }

    void printMatrixHead(const char *str, hpj::Matrix<float> &m) {

        printf("--------- %s ---------\n", str);
        if (m.Cols() > 1 && m.Rows() > 1) {
            int end_row = m.Rows()-1;
            int end_col = m.Cols()-1;
            printf("\tr[0,0], r[0,1] = %f, %f\n", m(0,0), m(0,1));
            printf("\tr[1,0], r[1,1] = %f, %f\n", m(1,0), m(1,1));
            printf("\tr[2,0], r[2,1] = %f, %f\n", m(2,0), m(2,1));

            printf("\tr[-3,-2], r[-3,-1] = %f, %f\n", m(end_row-2,end_col-1), m(end_row-2,end_col));
            printf("\tr[-2,-2], r[-2,-1] = %f, %f\n", m(end_row-1,end_col-1), m(end_row-1,end_col));
            printf("\tr[-1,-2], r[-1,-1] = %f, %f\n", m(end_row,end_col-1), m(end_row,end_col));
        } else if (m.Rows() == 1){
            printf("\tr[0,0], r[0,1], r[0,2] = %f, %f, %f\n", m(0,0), m(0,1), m(0,2));
        } else if (m.Cols() == 1){
            printf("\tr[0,0] = %f\n", m(0,0));
            printf("\tr[1,0] = %f\n", m(1,0));
            printf("\tr[2,0] = %f\n", m(2,0));
        }

    }


    void print_matrix(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C) {
        for (int i = 0; i < A.Rows(); ++i) {
            if (i < 2 || i > A.Rows() - 3) {
                float *presult = A.Row(i);
                for (int j = 0; j < A.Cols(); ++j) {
                    if (j < 2 || j > A.Cols() - 3)
                        std::cout << presult[j] << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        
        for (int i = 0; i < B.Rows(); ++i) {
            if (i < 2 || i > B.Rows() - 3) {
                float *presult = B.Row(i);
                for (int j = 0; j < B.Cols(); ++j) {
                    if (j < 2 || j > B.Cols() - 3)
                        std::cout << presult[j] << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
        
        for (int i = 0; i < C.Rows(); ++i) {
            if (i < 2 || i > C.Rows() - 3) {
                float *presult = C.Row(i);
                for (int j = 0; j < C.Cols(); ++j) {
                    if (j < 2 || j > C.Cols() - 3)
                        std::cout << presult[j] << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    void print_matrix(hpj::Matrix<float> src) {
        for (int i = 0; i < src.Rows(); ++i) {
            float *presult = src.Row(i);
            for (int j = 0; j < src.Cols(); ++j) {
                std::cout << presult[j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void sgemm_with_bias(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        MatMul_with_bias(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
        // print_matrix(A, B, C);
    }

    void sgemm_with_bias_qkv(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {

        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        bool wTrans = false;
        // A(128, 768), B(3*768, 768), C(3*128, 768) 
        int m = A.Rows();  // 128
        int k = A.Cols();  // 768
        int n = B.Cols();  // 768
        //printf("m = %d, k = %d, n = %d\n", m, k, n);
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

#if 0
        int lda = hiddenSize;
        int ldb = hiddenSize;
        int ldc = hiddenSize;

        int batch_stride_a = 0;
        int batch_stride_b = k*n;
        int batch_stride_c = m*n;
        int batch_stride_bias = n;

        int batch_src = 1;
        int batch_weights = 3;
        int batch_dst = 3;
        int batch_bias = 3;

        float scale = 1.f;

        BatchMatMul_with_stride_bias(eng, eng_stream, pA, pB1, pC1, pBias1, m, n, k, lda, ldb, ldc,
                batch_stride_a, batch_stride_b, batch_stride_c, batch_stride_bias,
                batch_src, batch_weights, batch_dst, batch_bias,
                scale, wTrans);
#else
        MatMul_with_bias(eng, eng_stream, pA, pB1, pBias1, pC1, m, n, k, wTrans);
        MatMul_with_bias(eng, eng_stream, pA, pB2, pBias2, pC2, m, n, k, wTrans);
        MatMul_with_bias(eng, eng_stream, pA, pB3, pBias3, pC3, m, n, k, wTrans);
#endif
    }


    void sgemm_with_bias_qkv_quant(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {

        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        bool wTrans = false;
        // A(128, 768), B(3*768, 768), C(3*128, 768) 
        int m = A.Rows();  // 128
        int k = A.Cols();  // 768
        int n = B.Cols();  // 768
        //printf("m = %d, k = %d, n = %d\n", m, k, n);
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
        //printf("qkv_src_max = %f\n", qkv_src_max);

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
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        MatMul_with_sum(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
        // print_matrix(A, B, C);
    }    

    void sgemm_with_sum_quant(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias,
                float src_scale, float weight_scale) {
        bool wTrans = (A.Cols() != B.Rows());
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        //MatMul_with_sum(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
        MatMul_with_sum_quant(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans, src_scale, weight_scale);
    }    

    void sgemm_with_sum_src_bf16(hpj::Matrix<bfloat16> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        bfloat16 *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        MatMul_with_sum_src_bf16(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
        // print_matrix(A, B, C);
    }    

    void sgemm_with_erf(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        const float factor = sqrt(0.5f);

        MatMul_with_erf(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
        // print_matrix(A, B, C);
    }    

    void sgemm_with_erf_quant(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = C.Data();
        float *pBias = bias.Data();

        const float factor = sqrt(0.5f);

        MatMul_with_erf_quant(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans, intermediate_SrcScale, intermediateWScale);
        // print_matrix(A, B, C);
    }    

    void sgemm_with_erf_dst_bf16(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<bfloat16> &C, hpj::Vector<float> &bias) {
        bool wTrans = (A.Cols() != B.Rows());
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
        int m = C.Rows();
        int k = A.Cols();
        int n = C.Cols();

        float *pA = A.Data();
        float *pB = B.Data();
        bfloat16 *pC = C.Data();
        float *pBias = bias.Data();

        const float factor = sqrt(0.5f);

        MatMul_with_erf_dst_bf16(eng, eng_stream, pA, pB, pBias, pC, m, n, k, wTrans);
        // print_matrix(A, B, C);
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

    void batchnorm(hpj::Matrix<float> &x, hpj::Vector<float> &gamma, hpj::Vector<float> &beta) {
        assert(x.Rows() == maxTokenSize);
        assert(x.Cols() == hiddenSize);

        float *pgamma = gamma.Data();
        float *pbeta = beta.Data();

        #pragma omp parallel for
        for (int i = 0; i < x.Rows(); ++i) {
            float sum = 0;
            float *px = x.Row(i);
            #pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                sum += px[j];
            }
            float mean = sum / hiddenSize;

            sum = 0;
            #pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                float delta = (px[j] - mean);
                sum += delta * delta;
            }
            float tmp = sum / hiddenSize + 9.999999960041972e-13;
            float rvariance = 1.0f / sqrt(tmp);

            #pragma omp simd
            for (int j = 0; j < x.Cols(); ++j) {
                px[j] = (px[j] - mean) * rvariance * pgamma[j] + pbeta[j];
            }
        }
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

// TODO(rfsaliev) clarify if below method is needed
    // ONLY for dimension 768
    // The first Batch MatMul inside self attention
/*    void batchMatMul_1(hpj::Matrix<float> &A, hpj::Matrix<float> &B, float *c_array[12]){
        #define GRP_COUNT 1
        MKL_INT    m[GRP_COUNT] = {maxTokenSize};
        MKL_INT    k[GRP_COUNT] = {64};
        MKL_INT    n[GRP_COUNT] = {maxTokenSize};
        
        MKL_INT    lda[GRP_COUNT] = {A.Stride()};
        MKL_INT    ldb[GRP_COUNT] = {B.Stride()};
        MKL_INT    ldc[GRP_COUNT] = {maxTokenSize};
        
        CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
        CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasTrans };
        
        float    alpha[GRP_COUNT] = {1.0};
        float    beta[GRP_COUNT] = {0.0};
        
        const MKL_INT    size_per_grp[GRP_COUNT] = {12};
        
        // Total number of multiplications: 12
        const float    *a_array[12], *b_array[12];
        for (int i = 0; i < 12; ++i) {
            a_array[i] = A.Data() + i * 64;
            b_array[i] = B.Data() + i * 64;
        }
        
        // Call cblas_sgemm_batch
        cblas_sgemm_batch (
                CblasRowMajor,
                transA,
                transB,
                m,
                n,
                k,
                alpha,
                a_array,
                lda,
                b_array,
                ldb,
                beta,
                c_array,
                ldc,
                GRP_COUNT,
                size_per_grp);
    }*/

    void batchMatMul_dnnl_1_with_scale_bias(hpj::Matrix<float> &A, hpj::Matrix<float> &B, float *c_array[12]){

        //bool wTrans = (A.Cols() != B.Rows());
        bool wTrans = true;
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
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

    void batchMatMul_dnnl_1(hpj::Matrix<float> &A, hpj::Matrix<float> &B, float *c_array[12]){

        //bool wTrans = (A.Cols() != B.Rows());
        bool wTrans = true;
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
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

        float *pA = A.Data();
        float *pB = B.Data();
        float *pC = c_array[0];
        
        //float *pBias = bias.Data();

        BatchMatMul_with_stride(eng, eng_stream, pA, pB, pC, m, n, k, lda, ldb, ldc, 
                            batch_stride_a, batch_stride_b, batch_stride_c, wTrans, batch);
    }

// TODO(rfsaliev) clarify if below method is needed
    // ONLY for dimension 768
    // The second Batch MatMul inside self attention
/*    void batchMatMul_2(float *a_array[12], hpj::Matrix<float> &B, hpj::Matrix<float> &C) {
        #define GRP_COUNT 1
        MKL_INT    m[GRP_COUNT] = {maxTokenSize};
        MKL_INT    k[GRP_COUNT] = {maxTokenSize};
        MKL_INT    n[GRP_COUNT] = {64};
        
        MKL_INT    lda[GRP_COUNT] = {maxTokenSize};
        MKL_INT    ldb[GRP_COUNT] = {B.Stride()};
        MKL_INT    ldc[GRP_COUNT] = {C.Stride()};
        
        CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
        CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasNoTrans };
        
        float    alpha[GRP_COUNT] = {1.0};
        float    beta[GRP_COUNT] = {0.0};
        
        const MKL_INT    size_per_grp[GRP_COUNT] = {12};
        
        // Total number of multiplications: 12
        float    *b_array[12], *c_array[12];
        for (int i = 0; i < 12; ++i) {
            b_array[i] = B.Data() + i * 64;
            c_array[i] = C.Data() + i * 64;
        }
        
        // Call cblas_sgemm_batch
        cblas_sgemm_batch (
                CblasRowMajor,
                transA,
                transB,
                m,
                n,
                k,
                alpha,
                (const float **)a_array,
                lda,
                (const float **)b_array,
                ldb,
                beta,
                c_array,
                ldc,
                GRP_COUNT,
                size_per_grp);
    }*/


    void batchMatMul_dnnl_2(float *a_array[12], hpj::Matrix<float> &B, hpj::Matrix<float> &C){

        //bool wTrans = (A.Cols() != B.Rows());
        bool wTrans = false;
        //printf("B.Rows = %d, B.Cols = %d\n", B.Rows(), B.Cols());
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

        
        //float *pBias = bias.Data();

        BatchMatMul_with_stride(eng, eng_stream, pA, pB, pC, m, n, k, lda, ldb, ldc, 
                        batch_stride_a, batch_stride_b, batch_stride_c, wTrans, batch);
    }

    // Add bias to matrix
    void biasAdd(hpj::Matrix<float> &m, hpj::Vector<float> &bias) {
        float *pbias = bias.Data();
        #pragma omp parallel for
        for (int i = 0; i < m.Rows(); ++i) {
            float *p = m.Row(i);
            #pragma omp simd
            for (int j = 0; j < m.Cols(); ++j) {
                p[j] += pbias[j];
            }
        }
    }

    void computeSoftmax_only_dnnl() {
        int rows = 12*maxTokenSize;
        int cols = maxTokenSize;
        float *pA = ctx.qk_result[0];

        Softmax(eng, eng_stream, pA, rows, cols);
    }

// TODO(rfsaliev) clarify if below methods are needed
/*    
    void computeSoftmax_only(int actualTokens) {

        #pragma omp parallel for
        for (int i = 0; i < 12; ++i) {
            float *pbuffer = ctx.exp_buffer[i];
            for (int row = 0; row < maxTokenSize; ++row) {
                float sum = 0;

                // min_val is used to avoid exp(x) = inf
                float min_val = std::numeric_limits<float>::max();
                #pragma omp simd
                for (int j = 0; j < actualTokens; ++j) {
                    if (ctx.qk_result[i][row*maxTokenSize+j] < min_val) {
                        min_val = ctx.qk_result[i][row*maxTokenSize+j];
                    }
                }
                if (min_val <= 0) {
                    min_val = 0;
                }
#ifdef __INTEL_COMPILER
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    //pbuffer[j] = exp(ctx.qk_result[i][row*maxTokenSize+j] * 0.125f + ctx.magic_value[j] - min_val);
                    pbuffer[j] = exp(ctx.qk_result[i][row*maxTokenSize+j] - min_val);
                    sum += pbuffer[j];
                }
#else
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    //pbuffer[j] = ctx.qk_result[i][row*maxTokenSize+j] * 0.125f + ctx.magic_value[j] - min_val;
                    pbuffer[j] = ctx.qk_result[i][row*maxTokenSize+j] - min_val;
                }
                vsExp(maxTokenSize, pbuffer, pbuffer);
                for (int j = 0; j < maxTokenSize; ++j) {
                    sum += pbuffer[j];
                }
#endif
                if (!isinf(sum)) {
                    #pragma omp simd
                    for (int j = 0; j < maxTokenSize; ++j) {
                        ctx.qk_result[i][row*maxTokenSize+j] = pbuffer[j] / sum;
                    }
                } else { // for the case of inf
                    int inf_num = 0;
                    memset(&ctx.qk_result[i][row*maxTokenSize+0], 0, sizeof(float) * maxTokenSize);
                    for (int j = 0; j < maxTokenSize; ++j) {
                        if (isinf(pbuffer[j])) { ctx.qk_result[i][row*maxTokenSize+j] = 1; inf_num += 1; } 
                    }
                    if (inf_num > 1) {
                        for (int j = 0; j < maxTokenSize; ++j) {
                            if (isinf(pbuffer[j])) { ctx.qk_result[i][row*maxTokenSize+j] = 1.0f / inf_num; } 
                        }
                    }
                }
            }
        }
    }
*/

    // input and output are both in qk_result
/*
    void computeSoftmax(int actualTokens) {
        for (int i = 0; i < actualTokens; ++i) { ctx.magic_value[i] = 0; }
        for (int i = actualTokens; i < maxTokenSize; ++i) { ctx.magic_value[i] = -10000; }

        #pragma omp parallel for
        for (int i = 0; i < 12; ++i) {
            float *pbuffer = ctx.exp_buffer[i];
            for (int row = 0; row < maxTokenSize; ++row) {
                float sum = 0;

                // min_val is used to avoid exp(x) = inf
                float min_val = std::numeric_limits<float>::max();
                #pragma omp simd
                for (int j = 0; j < actualTokens; ++j) {
                    if (ctx.qk_result[i][row*maxTokenSize+j] < min_val) {
                        min_val = ctx.qk_result[i][row*maxTokenSize+j];
                    }
                }
                if (min_val <= 0) {
                    min_val = 0;
                }
#ifdef __INTEL_COMPILER
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    pbuffer[j] = exp(ctx.qk_result[i][row*maxTokenSize+j] * 0.125f + ctx.magic_value[j] - min_val);
                    sum += pbuffer[j];
                }
#else
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    pbuffer[j] = ctx.qk_result[i][row*maxTokenSize+j] * 0.125f + ctx.magic_value[j] - min_val;
                }
                vsExp(maxTokenSize, pbuffer, pbuffer);
                for (int j = 0; j < maxTokenSize; ++j) {
                    sum += pbuffer[j];
                }
#endif
                if (!isinf(sum)) {
                    #pragma omp simd
                    for (int j = 0; j < maxTokenSize; ++j) {
                        ctx.qk_result[i][row*maxTokenSize+j] = pbuffer[j] / sum;
                    }
                } else { // for the case of inf
                    int inf_num = 0;
                    memset(&ctx.qk_result[i][row*maxTokenSize+0], 0, sizeof(float) * maxTokenSize);
                    for (int j = 0; j < maxTokenSize; ++j) {
                        if (isinf(pbuffer[j])) { ctx.qk_result[i][row*maxTokenSize+j] = 1; inf_num += 1; } 
                    }
                    if (inf_num > 1) {
                        for (int j = 0; j < maxTokenSize; ++j) {
                            if (isinf(pbuffer[j])) { ctx.qk_result[i][row*maxTokenSize+j] = 1.0f / inf_num; } 
                        }
                    }
                }
            }
        }
    }
*/

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

    bool init = false;
    //std::map<std::string, dnnl::impl::bfloat16_t*> weights_hub;

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


