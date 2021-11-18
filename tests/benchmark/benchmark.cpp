// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <limits>
#include <unistd.h>
#include "timer.h"
#include "my_types.h"
//#include "bert_layer_matmul_postop.h"
#include "bert_layer_quant_int8.h"

static const int LAYERS = 12;
static const int warmupTimes = 10;
static int benchmarkTimes = 1000;

static const int hiddenSize = 768;
static const int intermediateSize = 3072;
static const int attentionHeadNum = 12;

struct LayerWeights
{
  LayerWeights()
  {
    queryWeight = new float[hiddenSize * hiddenSize];
    keyWeight = new float[hiddenSize * hiddenSize];
    valueWeight = new float[hiddenSize * hiddenSize];
    attentionOutputWeight = new float[hiddenSize * hiddenSize];
    intermediateWeight = new float[hiddenSize * intermediateSize];
    outputWeight = new float[intermediateSize * hiddenSize];

    queryBias = new float[hiddenSize];
    keyBias = new float[hiddenSize];
    valueBias = new float[hiddenSize];
    attentionOutputBias = new float[hiddenSize];
    intermediateBias = new float[intermediateSize];
    outputBias = new float[hiddenSize];
    gamma1 = new float[hiddenSize];
    beta1 = new float[hiddenSize];
    gamma2 = new float[hiddenSize];
    beta2 = new float[hiddenSize];

    for (int i = 0; i < hiddenSize * hiddenSize; ++i)
    {
      queryWeight[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      keyWeight[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      valueWeight[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      attentionOutputWeight[i] = 1.0f * rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < hiddenSize * intermediateSize; ++i)
    {
      intermediateWeight[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      outputWeight[i] = 1.0f * rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < hiddenSize; ++i)
    {
      queryBias[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      keyBias[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      valueBias[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      attentionOutputBias[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      outputBias[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      gamma1[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      beta1[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      gamma2[i] = 1.0f * rand() / RAND_MAX - 0.5f;
      beta2[i] = 1.0f * rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < intermediateSize; ++i)
    {
      intermediateBias[i] = 1.0f * rand() / RAND_MAX - 0.5f;
    }
  }

  ~LayerWeights()
  {
    delete[] queryWeight;
    delete[] keyWeight;
    delete[] valueWeight;
    delete[] attentionOutputWeight;
    delete[] intermediateWeight;
    delete[] outputWeight;

    delete[] queryBias;
    delete[] keyBias;
    delete[] valueBias;
    delete[] attentionOutputBias;
    delete[] intermediateBias;
    delete[] outputBias;
    delete[] gamma1;
    delete[] beta1;
    delete[] gamma2;
    delete[] beta2;
  }

  float *queryWeight;
  float *keyWeight;
  float *valueWeight;
  float *attentionOutputWeight;
  float *intermediateWeight;
  float *outputWeight;

  float *queryBias;
  float *keyBias;
  float *valueBias;
  float *attentionOutputBias;
  float *intermediateBias;
  float *outputBias;
  float *gamma1;
  float *beta1;
  float *gamma2;
  float *beta2;
};

// MiniBatch = 1
void benchmarkMB1(int tokenSize, LayerWeights *weights, hpj::Matrix<float> &input)
{
  BertContext ctx;
  BertLayer *bert_layers[LAYERS];

  for (int i = 0; i < LAYERS; ++i)
  {
    bert_layers[i] = new BertLayer(ctx);
    bert_layers[i]->setWeights(weights[i].queryWeight, weights[i].queryBias,
                               weights[i].keyWeight, weights[i].keyBias,
                               weights[i].valueWeight, weights[i].valueBias,
                               weights[i].attentionOutputWeight, weights[i].attentionOutputBias,
                               weights[i].gamma1, weights[i].beta1,
                               weights[i].intermediateWeight, weights[i].intermediateBias,
                               weights[i].outputWeight, weights[i].outputBias,
                               weights[i].gamma2, weights[i].beta2);
  }

  std::vector<int> inputMask(ctx.maxTokenSize, 0);
  for (int i = 0; i < ctx.maxTokenSize; ++i) inputMask[i] = 1;
  ctx.setInputMask(inputMask.data());

  float totalTime = 0;
  for (int i = 0; i < warmupTimes + benchmarkTimes; ++i)
  {
    hpj::Matrix<float> *m_data = &input;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < LAYERS; ++i)
    {
      hpj::Matrix<float> &out = bert_layers[i]->forward(*m_data);
      m_data = &out;
    }

    if (i >= warmupTimes)
    {
      gettimeofday(&end, NULL);
      totalTime += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
    }
  }

  printf("LAST LOOP\n");
  sleep(2);
  {
    hpj::Matrix<float> *m_data = &input;

    for (int i = 0; i < LAYERS; ++i)
    {
      hpj::Matrix<float> &out = bert_layers[i]->forward(*m_data);
      m_data = &out;
    }
  }

  for (int i = 0; i < LAYERS; ++i)
  {
    delete bert_layers[i];
  }

  printf("Average Time: %.2f ms\n", totalTime / benchmarkTimes);
}

int main(int argc, char **argv)
{
  int tokenSize = 128;
  int batchSize = 1;

  if (argc > 1)
  {
    benchmarkTimes = atoi(argv[1]);
  }

  // Fake input
  hpj::Matrix<float> input;
  input.Resize(batchSize * tokenSize, hiddenSize);
  for (int i = 0; i < input.Rows(); ++i)
  {
    for (int j = 0; j < input.Cols(); ++j)
    {
      input(i, j) = 1.0f * rand() / RAND_MAX - 0.5f;
    }
  }

  // Fake weights
  LayerWeights weights[12];

  benchmarkMB1(tokenSize, weights, input);

  return 0;
}
