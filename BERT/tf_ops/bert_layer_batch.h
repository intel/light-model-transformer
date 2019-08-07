#ifndef BERT_LAYER_BATCH_H_
#define BERT_LAYER_BATCH_H_

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
#include "timer.h"

// int maxTokenSize = 128, int hiddenSize = 768, int intermediateSize = 3072, attentionHeadNum = 12
template<int maxTokenSize, int hiddenSize, int intermediateSize, int attentionHeadNum>
class BatchBertLayer
{
public:
    BatchBertLayer(int layerIdx) {
        this->layerIdx = layerIdx;
        this->batchSize = 0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) { num_threads = omp_get_num_threads(); }
        }
        
        qk_result = NULL;
        magic_value = NULL;

        // Preoare buffer of exp_buffer
        exp_buffer = new FloatPointer[num_threads];
        for (int i = 0; i < num_threads; ++i) {
            exp_buffer[i] = (float *)aligned_alloc(64, sizeof(float) * maxTokenSize);
        }

        // Preoare buffer of erf_buffer
        erf_buffer = new FloatPointer[num_threads];
        for (int i = 0; i < num_threads; ++i) {
            erf_buffer[i] = (float *)aligned_alloc(64, sizeof(float) * intermediateSize);
        }
    }

    virtual ~BatchBertLayer() {
        // exp_buffer, erf_buffer
        for (int i = 0; i < num_threads; ++i) {
            free(exp_buffer[i]);
            free(erf_buffer[i]);
        }
        delete[] exp_buffer;
        delete[] erf_buffer;
        exp_buffer = NULL;
        erf_buffer = NULL;

        // qk_result
        if (qk_result) {
            for (int i = 0; i < attentionHeadNum * batchSize; ++i) {
                free(qk_result[i]);
            }
            delete[] qk_result;
        }

        // magic_value
        if (magic_value) { free(magic_value); }
    }

    // When set batch size, may need to prepare some buffers
    void setBatchSize(int batchSize) {
        int preBatch = this->batchSize;
        if (preBatch == batchSize) {
            return;
        } else {
            this->batchSize = batchSize;
        }

        qkvMatMul.Resize(batchSize * maxTokenSize, hiddenSize*3);
        resultBuffer1.Resize(batchSize * maxTokenSize, hiddenSize);
        resultBuffer2.Resize(batchSize * maxTokenSize, hiddenSize);
        intermediateBuffer.Resize(batchSize * maxTokenSize, intermediateSize);

        // Preoare buffer of qk_result
        if (qk_result) {
            for (int i = 0; i < attentionHeadNum * preBatch; ++i) {
                free(qk_result[i]);
            }
            delete[] qk_result;
        }

        qk_result = new FloatPointer[attentionHeadNum * batchSize];
        for (int i = 0; i < attentionHeadNum * batchSize; ++i) {
            qk_result[i] = (float *)aligned_alloc(64, sizeof(float) * maxTokenSize * maxTokenSize);
        }

        // Preoare buffer of magic_value
        if (magic_value) { free(magic_value); }
        magic_value = (float *)aligned_alloc(64, sizeof(float) * batchSize * maxTokenSize);
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
    // input: (batchSize * maxTokenSize) x hidden_size
    // actualTokens: #tokens = maxTokenSize - padded_tokens
    hpj::Matrix<float> &forward(hpj::Matrix<float> &inputBuffer, std::vector<int> &actualTokens) {
        setBatchSize(actualTokens.size());
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
        //printf("qk_result[1]=%f,%f\n", qk_result[1][0], qk_result[1][1]);

        // Softmax
        computeSoftmax(actualTokens);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/attention/self/Softmax:\n", layerIdx);
        printf("%f, %f, ...\n", qk_result[0][0], qk_result[0][1]);
        printf("%f, %f, ...\n", qk_result[1][0], qk_result[1][1]);
#endif

        // BatchMatMul
        batchMatMul(qk_result, value, resultBuffer1);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/attention/self/Reshape_3:\n", layerIdx);
        dumpMatrix(resultBuffer1);
#endif

        // dense
        denseWithSum(resultBuffer1, attentionOutputWeight, attentionOutputBias, inputBuffer, resultBuffer2);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/attention/output/add:\n", layerIdx);
        dumpMatrix(resultBuffer2);
#endif

        // batchmorm
        batchnorm(resultBuffer2, gamma1, beta1);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/attention/output/LayerNorm/batchnorm/add_1:\n", layerIdx);
        dumpMatrix(resultBuffer2);
#endif
        
        // intermediate
        intermediate(resultBuffer2, intermediateBuffer);
#ifdef DEBUG
        printf("intermediate(bert/encoder/layer_%d/intermediate/dense/mul_1):\n", layerIdx);
        dumpMatrix(intermediateBuffer);
#endif

        // dense in output
        denseWithSum(intermediateBuffer, outputWeight, outputBias, resultBuffer2, resultBuffer1);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/output/add:\n", layerIdx);
        dumpMatrix(resultBuffer1);
#endif
        
        // batchnorm
        batchnorm(resultBuffer1, gamma2, beta2);
#ifdef DEBUG
        printf("bert/encoder/layer_%d/output/LayerNorm/batchnorm/add_1:\n", layerIdx);
        dumpMatrix(resultBuffer1);
#endif

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
        assert(x.Rows() == batchSize * maxTokenSize);
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

	/*void intermediate(hpj::Matrix<float> &input, hpj::Matrix<float> &output) {
        sgemm(input, intermediateWeight, output);

        float *pbias = intermediateBias.Data();
        float factor = 0.7978845608; // np.sqrt(2 / np.pi)

        #pragma omp parallel for
        for (int i = 0; i < output.Rows(); ++i) {
            int tid = omp_get_thread_num();
            float *pout = output.Row(i);
            #pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                float x = pout[j] + pbias[j];
                erf_buffer[tid][j] = x;
                pout[j] = factor * (x + 0.044715f * x * x * x);
            }
            vsTanh(output.Cols(), pout, pout);
            #pragma omp simd
            for (int j = 0; j < output.Cols(); ++j) {
                pout[j] = erf_buffer[tid][j] * 0.5f * (1 + pout[j]);
            }
        }
	}*/

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
    void batchMatMul(hpj::Matrix<float> &A, hpj::Matrix<float> &B, float **c_array){
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
        
        const MKL_INT    size_per_grp[GRP_COUNT] = {attentionHeadNum * batchSize};
        
        // Total number of multiplications: attentionHeadNum * batchSize
        const float **a_array = new ConstFloatPointer[attentionHeadNum * batchSize];
        const float **b_array = new ConstFloatPointer[attentionHeadNum * batchSize];
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < attentionHeadNum; ++i) {
                a_array[b*attentionHeadNum + i] = A.Row(b*maxTokenSize) + i * 64;
                b_array[b*attentionHeadNum + i] = B.Row(b*maxTokenSize) + i * 64;
            }
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
        delete[] a_array;
        delete[] b_array;
    }

    // ONLY for dimension 768
    // The second BatchMatMul inside self attention
    void batchMatMul(float *a_array[], hpj::Matrix<float> &B, hpj::Matrix<float> &C) {
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
        
        const MKL_INT    size_per_grp[GRP_COUNT] = {attentionHeadNum * batchSize};
        
        // Total number of multiplications: attentionHeadNum * batchSize
        const float **b_array = new ConstFloatPointer[attentionHeadNum * batchSize];
        float **c_array = new FloatPointer[attentionHeadNum * batchSize];
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < attentionHeadNum; ++i) {
                b_array[b*attentionHeadNum + i] = B.Row(b*maxTokenSize) + i * 64;
                c_array[b*attentionHeadNum + i] = C.Row(b*maxTokenSize) + i * 64;
            }
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

        delete[] b_array;
        delete[] c_array;
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
    void computeSoftmax(std::vector<int> &actualTokens) {
        #pragma omp parallel for
        for (int b = 0; b < batchSize; ++b) {
            memset(&magic_value[b * maxTokenSize], 0, sizeof(float) * actualTokens[b]);
            #pragma omp simd
            for (int i = actualTokens[b]; i < maxTokenSize; ++i) {
                magic_value[b * maxTokenSize + i] = -10000;
            }
        }


        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < attentionHeadNum; ++i) {
            int tid = omp_get_thread_num();
            float *pbuffer = exp_buffer[tid];
            float *result = qk_result[b*attentionHeadNum+i];

            for (int row = 0; row < maxTokenSize; ++row) {
                float sum = 0;

                // max_val is used to avoid exp(x) = inf
                float max_val = std::numeric_limits<float>::min();
                #pragma omp simd
                for (int j = 0; j < actualTokens[b]; ++j) {
                    if (result[j] > max_val) {
                        max_val = result[j];
                    }
                }
                max_val *= 0.125f;
#ifdef __INTEL_COMPILER
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    pbuffer[j] = exp(result[j] * 0.125f + magic_value[b * maxTokenSize + j] - max_val);
                    sum += pbuffer[j];
                }
#else
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    pbuffer[j] = result[j] * 0.125f + magic_value[b * maxTokenSize + j] - max_val;
                }
                vsExp(maxTokenSize, pbuffer, pbuffer);
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    sum += pbuffer[j];
                }
#endif
                float r_sum = 1.0f / sum;
                #pragma omp simd
                for (int j = 0; j < maxTokenSize; ++j) {
                    result[j] = pbuffer[j] * r_sum;
                }

                result += maxTokenSize;
            }
        }
        }
    }

private:
    // For debug usage
    int layerIdx;

    typedef float * FloatPointer;
    typedef const float * ConstFloatPointer;

    int batchSize;

    // Store the result of input*qkvWeight
    hpj::Matrix<float> qkvMatMul;
    // Buffer like the dimesion of 128x768
    hpj::Matrix<float> resultBuffer1, resultBuffer2;
    // Buffer to store the result of intermediate
    hpj::Matrix<float> intermediateBuffer;
    // Store the BatchMatMul result of query and key
    float **qk_result;
    // Store the result of exp for each line
    float **exp_buffer;
    // Temp buffer in intermediate
    float **erf_buffer;
    // Magic value: 0 or -10000
    float *magic_value;

    int num_threads;

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


