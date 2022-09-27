// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>
#include <random>

#include "bert_layer_quant_int8.h"
#include "bert_type_traits.h"
#include "dnnl_data.hpp"

static const int LAYERS = 12;
static const int warmupTimes = 10;
static int benchmarkTimes = 1000;

static const int hiddenSize = 768;
static const int intermediateSize = 3072;
static const int attentionHeadNum = 12;

struct LayerWeights
{
  LayerWeights(const dnnl::engine& eng, dnnl::stream& stm)
  {
    std::minstd_rand gen;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand = [&gen, &dist](){ return dist(gen); };

    std::vector<float> queryWeightBuffer(hiddenSize * hiddenSize);
    std::vector<float> keyWeightBuffer(hiddenSize * hiddenSize);
    std::vector<float> valueWeightBuffer(hiddenSize * hiddenSize);
    std::vector<float> attentionOutputWeightBuffer(hiddenSize * hiddenSize);
    std::vector<float> intermediateWeightBuffer(hiddenSize * intermediateSize);
    std::vector<float> outputWeightBuffer(hiddenSize * intermediateSize);
    std::vector<float> queryBiasBuffer(hiddenSize);
    std::vector<float> keyBiasBuffer(hiddenSize);
    std::vector<float> valueBiasBuffer(hiddenSize);
    std::vector<float> attentionOutputBiasBuffer(hiddenSize);
    std::vector<float> outputBiasBuffer(hiddenSize);
    std::vector<float> gamma1Buffer(hiddenSize);
    std::vector<float> beta1Buffer(hiddenSize);
    std::vector<float> gamma2Buffer(hiddenSize);
    std::vector<float> beta2Buffer(hiddenSize);
    std::vector<float> intermediateBiasBuffer(intermediateSize);

    std::generate(queryWeightBuffer.begin(), queryWeightBuffer.end(), rand);
    std::generate(keyWeightBuffer.begin(), keyWeightBuffer.end(), rand);
    std::generate(valueWeightBuffer.begin(), valueWeightBuffer.end(), rand);
    std::generate(attentionOutputWeightBuffer.begin(), attentionOutputWeightBuffer.end(), rand);
    std::generate(intermediateWeightBuffer.begin(), intermediateWeightBuffer.end(), rand);
    std::generate(outputWeightBuffer.begin(), outputWeightBuffer.end(), rand);
    std::generate(queryBiasBuffer.begin(), queryBiasBuffer.end(), rand);
    std::generate(keyBiasBuffer.begin(), keyBiasBuffer.end(), rand);
    std::generate(valueBiasBuffer.begin(), valueBiasBuffer.end(), rand);
    std::generate(attentionOutputBiasBuffer.begin(), attentionOutputBiasBuffer.end(), rand);
    std::generate(outputBiasBuffer.begin(), outputBiasBuffer.end(), rand);
    std::generate(gamma1Buffer.begin(), gamma1Buffer.end(), rand);
    std::generate(beta1Buffer.begin(), beta1Buffer.end(), rand);
    std::generate(gamma2Buffer.begin(), gamma2Buffer.end(), rand);
    std::generate(beta2Buffer.begin(), beta2Buffer.end(), rand);
    std::generate(intermediateBiasBuffer.begin(), intermediateBiasBuffer.end(), rand);
    
    queryWeight = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize * hiddenSize}, queryWeightBuffer.data());
    keyWeight = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize * hiddenSize}, keyWeightBuffer.data());
    valueWeight = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize * hiddenSize}, valueWeightBuffer.data());
    attentionOutputWeight = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize * hiddenSize}, attentionOutputWeightBuffer.data());
    intermediateWeight = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize * intermediateSize}, intermediateWeightBuffer.data());
    outputWeight = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize * intermediateSize}, outputWeightBuffer.data());
    queryBias = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, queryBiasBuffer.data());
    keyBias = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, keyBiasBuffer.data());
    valueBias = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, valueBiasBuffer.data());
    attentionOutputBias = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, attentionOutputBiasBuffer.data());
    outputBias = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, outputBiasBuffer.data());
    gamma1 = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, gamma1Buffer.data());
    beta1 = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, beta1Buffer.data());
    gamma2 = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, gamma2Buffer.data());
    beta2 = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{hiddenSize}, beta2Buffer.data());
    intermediateBias = dnnl_wrappers::CloneMemory(eng, stm, dnnl::memory::dims{intermediateSize}, intermediateBiasBuffer.data());
  }
public:
  dnnl::memory queryWeight;
  dnnl::memory keyWeight;
  dnnl::memory valueWeight;
  dnnl::memory attentionOutputWeight;
  dnnl::memory intermediateWeight;
  dnnl::memory outputWeight;
  dnnl::memory queryBias;
  dnnl::memory keyBias;
  dnnl::memory valueBias;
  dnnl::memory attentionOutputBias;
  dnnl::memory outputBias;
  dnnl::memory gamma1;
  dnnl::memory beta1;
  dnnl::memory gamma2;
  dnnl::memory beta2;
  dnnl::memory intermediateBias;
};

// MiniBatch = 1
template <bool do_quant, bool do_bf16>
void benchmark(int tokenSize, float *input, int batch = 1)
{
  using dt = dnnl::memory::data_type;

  auto ctx = std::make_shared<BertContext>(128, hiddenSize, intermediateSize, batch, LAYERS, do_quant, do_bf16);
  std::vector<std::unique_ptr<BertLayer>> bert_layers(LAYERS);
  std::vector<QuantizationFactors> quant_factors = {
      {-10.85244083404541015625, 4.14164829254150390625, -1.6212508678436279296875, 2.18305110931396484375, -64.5349578857421875, 9.17784881591796875, -0.16926576197147369384765625, 12.69039154052734375},
      {-10.01922702789306640625, 3.2598330974578857421875, -2.52011966705322265625, 3.17220592498779296875, -70.322662353515625, 4.564808368682861328125, -0.16925294697284698486328125, 10.93472957611083984375},
      {-11.37454319000244140625, 4.04611110687255859375, -2.5044767856597900390625, 3.4310567378997802734375, -56.21540069580078125, 5.208764553070068359375, -0.16948534548282623291015625, 72.20577239990234375},
      {-14.79791736602783203125, 4.259090423583984375, -2.8403589725494384765625, 3.91925144195556640625, -93.42563629150390625, 5.099577426910400390625, -0.1689991652965545654296875, 9.5706195831298828125},
      {-13.21285343170166015625, 4.449753284454345703125, -3.1772515773773193359375, 4.3330135345458984375, -101.334869384765625, 5.41256046295166015625, -0.16838109493255615234375, 10.64498996734619140625},
      {-13.93945217132568359375, 5.1448192596435546875, -2.5481836795806884765625, 3.48368167877197265625, -91.05278778076171875, 5.9057769775390625, -0.16948328912258148193359375, 12.6811923980712890625},
      {-14.12649059295654296875, 5.23845577239990234375, -2.814735889434814453125, 3.2215893268585205078125, -89.623870849609375, 6.68107700347900390625, -0.16898013651371002197265625, 11.01731777191162109375},
      {-13.5746974945068359375, 4.71494960784912109375, -2.7004568576812744140625, 3.2631299495697021484375, -87.90279388427734375, 7.388260364532470703125, -0.16951541602611541748046875, 8.03197765350341796875},
      {-15.597011566162109375, 6.920653820037841796875, -3.0222375392913818359375, 3.777666568756103515625, -83.6142730712890625, 10.2494525909423828125, -0.1686449944972991943359375, 23.9402790069580078125},
      {-15.88373565673828125, 10.81757640838623046875, -2.6777179241180419921875, 3.3885133266448974609375, -48.061458587646484375, 16.7345333099365234375, -0.156786620616912841796875, 92.52396392822265625},
      {-18.6183719635009765625, 11.54715251922607421875, -2.11896610260009765625, 3.066336154937744140625, -41.8497314453125, 19.4496479034423828125, -0.16698478162288665771484375, 141.4157867431640625},
      {-23.8061676025390625, 11.55181217193603515625, -2.552584171295166015625, 3.7034885883331298828125, -36.45532989501953125, 16.997623443603515625, -0.16963402926921844482421875, 8.112117767333984375},
  };

  std::vector<LayerWeights> weights;
  weights.reserve(LAYERS);

  for (int i = 0; i < LAYERS; ++i)
  {
    weights.emplace_back(ctx->dnnl_context.getEngine(), ctx->dnnl_context.getEngineStream());
    bert_layers[i] = std::make_unique<BertLayer>(ctx);
    bert_layers[i]->setWeights(weights[i].queryWeight, weights[i].queryBias,
                               weights[i].keyWeight, weights[i].keyBias,
                               weights[i].valueWeight, weights[i].valueBias,
                               weights[i].attentionOutputWeight, weights[i].attentionOutputBias,
                               weights[i].gamma1, weights[i].beta1,
                               weights[i].intermediateWeight, weights[i].intermediateBias,
                               weights[i].outputWeight, weights[i].outputBias,
                               weights[i].gamma2, weights[i].beta2, quant_factors[i]);
  }

  using dims = dnnl::memory::dims;
  dnnl::memory input_mask{dnnl::memory::desc{{ctx->batch_, 1, 1, ctx->maxTokenSize}, dt::f32, dims{}}, ctx->dnnl_context.getEngine()};

  using duration = std::chrono::steady_clock::duration;
  std::vector<duration> compute_times;
  compute_times.reserve(benchmarkTimes);

  for (int i = 0; i < warmupTimes + benchmarkTimes; ++i)
  {
    dnnl::memory::dims dims{batch * 128, 768};
    auto buffer = dnnl_wrappers::AttachMemory(ctx->dnnl_context.getEngine(), dims, input, false);

    auto start = std::chrono::steady_clock::now();
    for (int j = 0; j < LAYERS; ++j)
    {
      bert_layers[j]->forward(buffer, input_mask);
    }
    auto end = std::chrono::steady_clock::now();

    if (i >= warmupTimes)
    {
      compute_times.push_back(end - start);
    }
  }

  using namespace std::chrono_literals;

  auto total_samples = static_cast<double>(compute_times.size() * batch);

  // We want to have time in seconds but keep the fraction part for precision.
  std::chrono::duration<double, std::ratio<1, 1>> total_time_s = std::accumulate(std::begin(compute_times), std::end(compute_times), duration{0ns});

  // No duration_cast needed for floating point durations.
  std::chrono::duration<double, std::milli> average_time_ms = total_time_s / total_samples;

  auto throughput_per_s = total_samples / total_time_s.count();


  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << "Average Time: " << average_time_ms.count() << " ms" << std::endl;
  ss << std::fixed << std::setprecision(2) << "Average Throughput: " << throughput_per_s << " samples/s" << std::endl;
  std::cout << ss.str();
}

int main(int argc, char **argv)
{
  int tokenSize = 128;
  int batchSize = 1;

try {
  if (argc > 1)
  {
    benchmarkTimes = std::stoi(argv[1]);
  }
  if (benchmarkTimes < 1)
  {
    throw std::invalid_argument("Amount of times benchmark is run cannot be less than 1");
  }

  if (argc > 2)
  {
    batchSize = std::stoi(argv[2]);
  }
  if (batchSize < 1)
  {
    throw std::invalid_argument("Batch size cannot be less than 1");
  }

  // Fake input
  std::vector<float> input(batchSize * tokenSize * hiddenSize);
  std::minstd_rand gen; //faster than MT
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::generate(input.begin(), input.end(), [&gen, &dist](){ return  dist(gen); });

  // Get BertLayer mode from command line args to let CI decide what to run.
  // Defaults to FP32.
  std::vector<std::string> flags;
  for (int i = 1; i < argc; ++i)
  {
      flags.push_back(argv[i]);
  }
  bool do_quantization = std::find(begin(flags), end(flags), "--quantization") != end(flags);
  bool do_bfloat16 = std::find(begin(flags), end(flags), "--bfloat16") != end(flags);
  std::cout << "BertLayer mode: " << (do_quantization ? "int8 quantization" : "no quantization") << ", "
                                  << (do_bfloat16 ? "bfloat16" : "fp32") << std::endl;
  
  // Ugly, but we don't know the BertLayer mode at compile time.
  if (do_quantization)
  {
    if (do_bfloat16)
    {
        benchmark<true, true>(tokenSize, input.data(), batchSize);
    }
    else
    {
        benchmark<true, false>(tokenSize, input.data(), batchSize);
    }
  }
  else
  {
    if (do_bfloat16)
    {
        benchmark<false, true>(tokenSize, input.data(), batchSize);
    }
    else
    {
        benchmark<false, false>(tokenSize, input.data(), batchSize);
    }
  }

} catch (const std::invalid_argument& ia) {
  std::cerr << "Caught invalid argument exception: " << ia.what() << std::endl;
  return 1;
} catch (const std::out_of_range& e) {
  std::cerr << "Caught out of range exception: " << e.what() << std::endl;
  return 1;
} catch (const dnnl::error& e) {
  std::cerr << "Caught oneDNN error: " << e.what() << std::endl;
  return 1;
} catch (const std::exception& e) {
  std::cerr << "Caught exception: " << e.what() << std::endl;
  return 1;
} catch (...) {
  std::cerr << "Caught unknown exception." << std::endl;
  return 1;
}

  return 0;
}
