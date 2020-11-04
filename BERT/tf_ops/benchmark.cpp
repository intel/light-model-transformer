#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <limits>
#include "timer.h"
#include "my_types.h"
#include "bert_layer_batch.h"
#include "bert_layer_mb1_dynamic_tokens.h"

static const int LAYERS = 12;
static const int warmupTimes = 10;
static const int benchmarkTimes = 1000;

static const int hiddenSize = 768;
static const int intermediateSize = 3072;
static const int attentionHeadNum = 12;

#define BENCH_BATCH_BERT(tokenSize) \
  BatchBertLayer<tokenSize, hiddenSize, intermediateSize, attentionHeadNum> *batch_bert_layers[LAYERS]; \
  for (int i = 0; i < LAYERS; ++i) { \
    batch_bert_layers[i] = new BatchBertLayer<tokenSize, hiddenSize, intermediateSize, attentionHeadNum>(i); \
    batch_bert_layers[i]->setWeights(weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights); \
  } \
  \
  float totalTime = 0; \
  for (int i = 0; i < warmupTimes + benchmarkTimes; ++i) { \
    std::vector<int> actual_tokens(batchSize, tokenSize); \
    hpj::Matrix<float> *m_data = &input; \
    struct timeval start, end; \
    gettimeofday(&start, NULL); \
    for (int i = 0; i < LAYERS; ++i) { \
      hpj::Matrix<float> &out = batch_bert_layers[i]->forward(*m_data, actual_tokens); \
      m_data = &out; \
    } \
    if (i >= warmupTimes) { \
      gettimeofday(&end, NULL);  \
      totalTime += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f; \
    } \
  } \
  \
  for (int i = 0; i < LAYERS; ++i) { \
    delete batch_bert_layers[i]; \
  } \
  \
  printf("Average Time: %.2f ms\n", totalTime / benchmarkTimes);

void benchmarkBatch(int tokenSize, int batchSize, float *weights, hpj::Matrix<float> &input) {
  if (tokenSize == 16) {
    BENCH_BATCH_BERT(16)
  } else if (tokenSize == 32) {
    BENCH_BATCH_BERT(32)
  } else if (tokenSize == 64) {
    BENCH_BATCH_BERT(64)
  } else if (tokenSize == 128) {
    BENCH_BATCH_BERT(128)
  } else if (tokenSize == 256) {
    BENCH_BATCH_BERT(256)
  } else if (tokenSize == 512) {
    BENCH_BATCH_BERT(512)
  }
}

// MiniBatch = 1
void benchmarkMB1(int tokenSize, float *weights, hpj::Matrix<float> &input) {
  BertLayer *bert_layers[LAYERS];
  for (int i = 0; i < LAYERS; ++i) {
    bert_layers[i] = new BertLayer(i, tokenSize, hiddenSize, intermediateSize);
    bert_layers[i]->setWeights(weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights, weights);
  }

  float totalTime = 0;
  for (int i = 0; i < warmupTimes + benchmarkTimes; ++i) {
    hpj::Matrix<float> *m_data = &input;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (int i = 0; i < LAYERS; ++i) {
      hpj::Matrix<float> &out = bert_layers[i]->forward(*m_data, tokenSize, tokenSize);
      m_data = &out;
    }
    if (i >= warmupTimes) {
      gettimeofday(&end, NULL); 
      totalTime += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
    }
  }

  for (int i = 0; i < LAYERS; ++i) {
    delete bert_layers[i];
  }

  printf("Average Time: %.2f ms\n", totalTime / benchmarkTimes);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    printf("Usage: %s tokenSize batchSize\n", argv[0]);
    exit(-1);
  }

  int tokenSize = atoi(argv[1]);
  int batchSize = atoi(argv[2]);

  // Fake input
  hpj::Matrix<float> input;
  input.Resize(batchSize * tokenSize, hiddenSize);
  for (int i = 0; i < input.Rows(); ++i) {
    for (int j = 0; j < input.Cols(); ++j) {
      input(i, j) = 1.0f;
    }
  }

  // Fake weights
  float *weights = new float[hiddenSize * intermediateSize];
  for (int i = 0; i < hiddenSize * intermediateSize; ++i) {
    weights[i] = 1.0f * rand() / RAND_MAX - 0.5f;
  }

  if (batchSize == 1) {
    benchmarkMB1(tokenSize, weights, input);
  } else {
    benchmarkBatch(tokenSize, batchSize, weights, input);
  }

  // Clean up
  delete[] weights;

  return 0;
}
