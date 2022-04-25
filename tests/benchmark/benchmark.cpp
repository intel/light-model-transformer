// Copyright (C) 2021 Intel Corporation
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

static const int LAYERS = 12;
static const int warmupTimes = 10;
static int benchmarkTimes = 1000;

static const int hiddenSize = 768;
static const int intermediateSize = 3072;
static const int attentionHeadNum = 12;

struct LayerWeights
{
  LayerWeights()
    : queryWeight(hiddenSize * hiddenSize)
    , keyWeight(hiddenSize * hiddenSize)
    , valueWeight(hiddenSize * hiddenSize)
    , attentionOutputWeight(hiddenSize * hiddenSize)
    , intermediateWeight(hiddenSize * intermediateSize)
    , outputWeight(hiddenSize * intermediateSize)
    , queryBias(hiddenSize)
    , keyBias(hiddenSize)
    , valueBias(hiddenSize)
    , attentionOutputBias(hiddenSize)
    , outputBias(hiddenSize)
    , gamma1(hiddenSize)
    , beta1(hiddenSize)
    , gamma2(hiddenSize)
    , beta2(hiddenSize)
    , intermediateBias(intermediateSize)
  {
    std::minstd_rand gen;
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    auto rand = [&gen, &dist](){ return dist(gen); };

    std::generate(queryWeight.begin(), queryWeight.end(), rand);
    std::generate(keyWeight.begin(), keyWeight.end(), rand);
    std::generate(valueWeight.begin(), valueWeight.end(), rand);
    std::generate(attentionOutputWeight.begin(), attentionOutputWeight.end(), rand);

    std::generate(intermediateWeight.begin(), intermediateWeight.end(), rand);
    std::generate(outputWeight.begin(), outputWeight.end(), rand);

    std::generate(queryBias.begin(), queryBias.end(), rand);
    std::generate(keyBias.begin(), keyBias.end(), rand);
    std::generate(valueBias.begin(), valueBias.end(), rand);
    std::generate(attentionOutputBias.begin(), attentionOutputBias.end(), rand);
    std::generate(outputBias.begin(), outputBias.end(), rand);
    std::generate(gamma1.begin(), gamma1.end(), rand);
    std::generate(beta1.begin(), beta1.end(), rand);
    std::generate(gamma2.begin(), gamma2.end(), rand);
    std::generate(beta2.begin(), beta2.end(), rand);

    std::generate(intermediateBias.begin(), intermediateBias.end(), rand);
  }

  std::vector<float> queryWeight;
  std::vector<float> keyWeight;
  std::vector<float> valueWeight;
  std::vector<float> attentionOutputWeight;

  std::vector<float> intermediateWeight;
  std::vector<float> outputWeight;

  std::vector<float> queryBias;
  std::vector<float> keyBias;
  std::vector<float> valueBias;
  std::vector<float> attentionOutputBias;
  std::vector<float> outputBias;
  std::vector<float> gamma1;
  std::vector<float> beta1;
  std::vector<float> gamma2;
  std::vector<float> beta2;

  std::vector<float> intermediateBias;
};

// MiniBatch = 1
// template <class InputT, class BatchInputT>
// void benchmarkMB1(int tokenSize, LayerWeights *weights, float *input)
// {
//   using BertContextT = BertContext<InputT, BatchInputT>;

template <bool do_quant, bool do_bf16>
void benchmarkMB1(int tokenSize, LayerWeights *weights, float *input)
{
  using InputT = typename use_quantization<do_quant>::type;
  using BatchInputT = typename use_bfloat16<do_bf16>::type;
  using BertContextT = BertContext<InputT, BatchInputT>;

  BertContextT ctx;
  BertLayer<BertContextT> *bert_layers[LAYERS];
  Layer_minmax bert_layers_minmax[12] = {
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



  for (int i = 0; i < LAYERS; ++i)
  {
    bert_layers[i] = new BertLayer<BertContextT>(ctx);
    bert_layers[i]->setWeights(weights[i].queryWeight.data(), weights[i].queryBias.data(),
                               weights[i].keyWeight.data(), weights[i].keyBias.data(),
                               weights[i].valueWeight.data(), weights[i].valueBias.data(),
                               weights[i].attentionOutputWeight.data(), weights[i].attentionOutputBias.data(),
                               weights[i].gamma1.data(), weights[i].beta1.data(),
                               weights[i].intermediateWeight.data(), weights[i].intermediateBias.data(),
                               weights[i].outputWeight.data(), weights[i].outputBias.data(),
                               weights[i].gamma2.data(), weights[i].beta2.data(),
                               bert_layers_minmax[i]);
  }

  std::vector<int> inputMask(ctx.maxTokenSize, 1);
  ctx.setInputMask(inputMask.data());

  using duration = std::chrono::steady_clock::duration;
  std::vector<duration> compute_times;
  compute_times.reserve(benchmarkTimes);

  for (int i = 0; i < warmupTimes + benchmarkTimes; ++i)
  {
    dnnl::memory::dims dims{128, 768};
    auto buffer = dnnl_wrappers::AttachMemory(ctx.dnnl_context.getEngine(), dims, input, false);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < LAYERS; ++i)
    {
      bert_layers[i]->forward(buffer);
    }
    auto end = std::chrono::steady_clock::now();

    if (i >= warmupTimes)
    {
      compute_times.push_back(end - start);
    }
  }

  using namespace std::chrono_literals;
  auto total_time = std::accumulate(std::begin(compute_times), std::end(compute_times), duration{0ns});
  auto average_time_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(total_time).count() / compute_times.size();

  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << "Average Time: " << average_time_ms << " ms" << std::endl;
  std::cout << ss.str();

  for (int i = 0; i < LAYERS; ++i)
  {
      delete bert_layers[i];
  }
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
  // Fake input
  std::vector<float> input(batchSize * tokenSize * hiddenSize);
  std::minstd_rand gen; //faster than MT
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::generate(input.begin(), input.end(), [&gen, &dist](){ return  dist(gen); });

  // Fake weights
  LayerWeights weights[12];

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
        benchmarkMB1<true, true>(tokenSize, weights, input.data());
    }
    else
    {
        benchmarkMB1<true, false>(tokenSize, weights, input.data());
    }
  }
  else
  {
    if (do_bfloat16)
    {
        benchmarkMB1<false, true>(tokenSize, weights, input.data());
    }
    else
    {
        benchmarkMB1<false, false>(tokenSize, weights, input.data());
    }
  }

} catch (const std::invalid_argument& ia) {
  std::cerr << "Caught invalid argument exception: " << ia.what() << std::endl;
  return 1;
} catch (const std::out_of_range& e) {
  std::cerr << "Caught out of range exception: " << e.what() << std::endl;
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
