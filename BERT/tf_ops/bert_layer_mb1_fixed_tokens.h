#ifndef BERT_LAYER_H_
#define BERT_LAYER_H_

#include <new>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include <iostream>
#include <immintrin.h>
#include "my_types.h"
//#include "timer.h"

class BertLayer
{
public:
    // hiddenSize 768 隐层神经元、隐藏单元数
    // intermediateSize 3072 feed-forward/filter size 升维维度 4*hiddensize 
    BertLayer(int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072) {
        this->maxTokenSize = maxTokenSize;
        this->hiddenSize = hiddenSize;
        this->intermediateSize = intermediateSize;

        qkvMatMul.Resize(maxTokenSize, hiddenSize*3);
        resultBuffer1.Resize(maxTokenSize, hiddenSize);
        resultBuffer2.Resize(maxTokenSize, hiddenSize);
        intermediateBuffer.Resize(maxTokenSize, intermediateSize);

        for (int i = 0; i < 12; ++i) {
            qk_result[i] = (float *)aligned_alloc(64, sizeof(float) * maxTokenSize * maxTokenSize);
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

    virtual ~BertLayer() {
        for (int i = 0; i < 12; ++i) {
            free(qk_result[i]);
            free(exp_buffer[i]);
            qk_result[i] = NULL;
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
        
        tmp.Resize(hiddenSize, hiddenSize * 3);
        copyWeights(tmp, 0, hiddenSize, _queryWeight);
        copyWeights(tmp, hiddenSize, hiddenSize*2, _keyWeight);
        copyWeights(tmp, hiddenSize*2, hiddenSize*3, _valueWeight);
        copyTransposed(qkvWeight, tmp);
        /*
        qkvWeight.Resize(hiddenSize, hiddenSize * 3);
        copyWeights(qkvWeight, 0, hiddenSize, _queryWeight);
        copyWeights(qkvWeight, hiddenSize, hiddenSize*2, _keyWeight);
        copyWeights(qkvWeight, hiddenSize*2, hiddenSize*3, _valueWeight);
        */

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

        // gamma and beta for batchnorm after self attention
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

        // gamma and beta for the last batchnorm
        gamma2.Resize(hiddenSize);
        beta2.Resize(hiddenSize);
        memcpy(gamma2.Data(), _gamma2, sizeof(float) * hiddenSize);
        memcpy(beta2.Data(), _beta2, sizeof(float) * hiddenSize);
    }


    // Do the forward computing for the whole BERT layer
    // input: maxTokenSize x hidden_size
    // actualTokens: #tokens = maxTokenSize - padded_tokens
    hpj::Matrix<float> &forward(hpj::Matrix<float> &inputBuffer, int actualTokens) {
        // Query, Key, Value computed together
        sgemm(inputBuffer, qkvWeight, qkvMatMul);
        biasAdd(qkvMatMul, qkvBias);
        //dumpMatrix(qkvMatMul);

        // BatchMatMul
        hpj::Matrix<float> query(qkvMatMul, 0, qkvMatMul.Rows(), 0, hiddenSize);
        hpj::Matrix<float> key(qkvMatMul, 0, qkvMatMul.Rows(), hiddenSize, hiddenSize);
        hpj::Matrix<float> value(qkvMatMul, 0, qkvMatMul.Rows(), hiddenSize*2, hiddenSize);
        batchMatMul(query, key, qk_result);
        //printf("qk_result[0]=%f,%f\n", qk_result[0][0], qk_result[0][1]);

        // Softmax
        computeSoftmax(actualTokens);
        //printf("after softmax, qk_result[0]=%f,%f\n", qk_result[0][0], qk_result[0][1]);

        // BatchMatMul
        batchMatMul(qk_result, value, resultBuffer1);
        //printf("batchMatMul:\n");
        //dumpMatrix(resultBuffer1);

        // dense
        denseWithSum(resultBuffer1, attentionOutputWeight, attentionOutputBias, inputBuffer, resultBuffer2);
        //printf("denseWithSum:\n");
        //dumpMatrix(resultBuffer2);

        // batchmorm
        batchnorm(resultBuffer2, gamma1, beta1);
        //printf("batchnorm:\n");
        //dumpMatrix(resultBuffer2);
        
        // intermediate
        intermediate(resultBuffer2, intermediateBuffer);
        //printf("intermediate:\n");
        //dumpMatrix(intermediateBuffer);

        // dense in output
        denseWithSum(intermediateBuffer, outputWeight, outputBias, resultBuffer2, resultBuffer1);
        //dumpMatrix(resultBuffer1);
        
        // batchnorm
        batchnorm(resultBuffer1, gamma2, beta2);
        //dumpMatrix(resultBuffer1);

        return resultBuffer1;
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

    // C = A * B
    // bTranspose: B need to be transposed or not
    void sgemm(hpj::Matrix<float> &A, hpj::Matrix<float> &B, hpj::Matrix<float> &C) {
        bool bTranspose = (A.Cols() != B.Rows());
        int m = A.Rows();
        int k = A.Cols();
        int n = (bTranspose ? B.Rows() : B.Cols());
        float alpha = 1;
        float beta = 0;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, (bTranspose ? CblasTrans : CblasNoTrans), 
                    m, n, k, alpha,
                    A.Data(), A.Stride(), 
                    B.Data(), B.Stride(), beta,
                    C.Data(), C.Stride());
    }

    // result = x * weight + bias + input
    void denseWithSum(hpj::Matrix<float> &x, hpj::Matrix<float> &weight, hpj::Vector<float> &bias, hpj::Matrix<float> &input, hpj::Matrix<float> &result) {
        assert(input.Rows() == result.Rows());
        assert(input.Cols() == result.Cols());

        sgemm(x, weight, result);

        float *pbias = bias.Data();

        #pragma omp parallel for
        for (int i = 0; i < result.Rows(); ++i) {
            float *presult = result.Row(i);
            float *pinput = input.Row(i);
            #pragma omp simd
            for (int j = 0; j < result.Cols(); ++j) {
                presult[j] += pinput[j] + pbias[j];
            }
        }
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

    void intermediate(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
        sgemm(input, intermediateWeight, output);

        float *pbias = intermediateBias.Data();
        const float factor = sqrt(0.5f);
        const float scale = 0.5f / factor;

#ifdef __INTEL_COMPILER
        #pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            float *pout = output.Row(i);
            #pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                float with_bias = pout[j] + pbias[j];
                pout[j] = with_bias * 0.5f * (erf(with_bias * factor) + 1);
            }
        }
#else
        #pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            int tid = omp_get_thread_num();
            float *pout = output.Row(i);
            #pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                pout[j] = (pout[j] + pbias[j]) * factor;
            }
            vsErf(output.Cols(), pout, erf_buffer[tid]);
            #pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                pout[j] = pout[j] * scale * (erf_buffer[tid][j] + 1);
            }
        }
#endif
    }

    // ONLY for dimension 768
    // The first BatchMatMul inside self attention
    void batchMatMul(hpj::Matrix<float> &A, hpj::Matrix<float> &B, float *c_array[12]){
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
    }

    // ONLY for dimension 768
    // The second BatchMatMul inside self attention
    void batchMatMul(float *a_array[12], hpj::Matrix<float> &B, hpj::Matrix<float> &C) {
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

    // input and output are both in qk_result
    void computeSoftmax(int actualTokens) {
        for (int i = 0; i < actualTokens; ++i) { magic_value[i] = 0; }
        for (int i = actualTokens; i < maxTokenSize; ++i) { magic_value[i] = -10000; }

        #pragma omp parallel for
        for (int i = 0; i < 12; ++i) {
            float *pbuffer = exp_buffer[i];
            for (int row = 0; row < maxTokenSize; ++row) {
                float sum = 0;

                // max_val is used to avoid exp(x) = inf
                float max_val = std::numeric_limits<float>::min();
                #pragma omp simd
                for (int j = 0; j < actualTokens; ++j) {
                    if (qk_result[i][row*maxTokenSize+j] > max_val) {
                        max_val = qk_result[i][row*maxTokenSize+j];
                    }
                }
                max_val *= 0.125f;
#ifdef __INTEL_COMPILER
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    pbuffer[j] = exp(qk_result[i][row*maxTokenSize+j] * 0.125f + magic_value[j] - max_val);
                    sum += pbuffer[j];
                }
#else
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    pbuffer[j] = qk_result[i][row*maxTokenSize+j] * 0.125f + magic_value[j] - max_val;
                }
                vsExp(maxTokenSize, pbuffer, pbuffer);
                for (int j = 0; j < maxTokenSize; ++j) {
                    sum += pbuffer[j];
                }
#endif
                float r_sum = 1.0f / sum;
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    qk_result[i][row*maxTokenSize+j] = pbuffer[j] * r_sum;
                }
            }
        }
    }

private:
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
    float *qk_result[12];
    // Store the result of exp for each line
    float *exp_buffer[12];
    // Magic value: 0 or -10000
    float *magic_value;

    int num_threads;
#ifndef __INTEL_COMPILER 
    float **erf_buffer;
#endif

    // Merged query, key, value weighs
    hpj::Matrix<float> qkvWeight;
    // Merged query, key, value bias
    hpj::Vector<float> qkvBias;

    hpj::Matrix<float> attentionOutputWeight;
    hpj::Vector<float> attentionOutputBias;

    // batchnorm param
    hpj::Vector<float> gamma1, beta1;
    hpj::Vector<float> gamma2, beta2;

    hpj::Matrix<float> intermediateWeight;
    hpj::Vector<float> intermediateBias;

    hpj::Matrix<float> outputWeight;
    hpj::Vector<float> outputBias;
};

#endif


