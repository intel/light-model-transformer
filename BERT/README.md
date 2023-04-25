# BERT model optimization

BERT model optimization is an open-source optimization for BERT language processing model.
The optimization is based on [Bfloat16 Optimization Boosts Alibaba Cloud BERT Model Performance](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/alibaba-blog.html).  
Furthermore, it utilizes  [Intel® oneAPI Deep Neural Network Library (oneDNN)](https://github.com/oneapi-src/oneDNN) to obtain additional performance gains.  
BERT model optimization is split into two parts, model modifier, which modifies the model to use a custom operator and a custom operator which utilizes oneDNN.  
Currently HuggingFace BERT models (PyTorch and TensorFlow backends) and models built using TensorFlow 1.x and 2.x are supported.
We provide the way to modify some TensorFLow models from TFhub, google-research/bert and Hugging Face.
If you wish to modify your custom TensorFlow model, we provide the step by step guide how to do it. Please check our [README](util/README.md) page.

## Table of contents

* [Requirements for building from source](#requirements-for-building-from-source)
* [Building from source](#building-from-source)
* [Getting started](#getting-started)
* [Samples](#samples)
* [License](#license)
* [Features and known issues](#features-and-known-issues)
* [Support](#support)

## Requirements for building from source

BERT model optimization supports systems meeting the following requirements:

* [oneDNN Library](https://github.com/oneapi-src/oneDNN) 3.1 or later
* C++ compiler
* [CMake](https://cmake.org/download/)
* Linux based operating system 

## Building from source

1. Clone and build:

    ```sh
    git clone https://github.com/intel/light-model-transformer
    cd light-model-transformer/BERT
    mkdir build
    cd build
    source /opt/intel/oneapi/setvars.sh # Make sure CMake can find oneDNN
    cmake .. -DBACKENDS="TF\;PT" # Use TF (Tensorflow), PT (PyTorch) or both, based on which frameworks you wish to use.
    cmake --build . -j 8
    ```

2. Run benchmark: `tests/benchmark/benchmark`

## Getting started

For the currently supported use cases, short tutorials on usage are provided.
All of them require built from source the BERT Operator (BertOp), refer to [Building from source](#building-from-source)

* [tensorflow 1.x](tests/tf1_ops_accuracy/README.md) (*Tested on TF v.1.15*)
* [tensorflow 2.x](tests/tf2_ops_accuracy/README.md) (*Tested on TF v.2.5, v.2.9, v.2.12*)
* [tensorflow 2.x without using the model_modifier module](tests/tf2_no_model_modifier/README.md) (only HuggingFace models are currently supported)
* [pytorch](tests/pytorch/README.md) (only HuggingFace models are currently supported)
* [Model Zoo for Intel® Architecture](tests/model_zoo/README.md)

## Samples

There are scripts which demonstrate BertOp integration capabilities:

* [TensorFlow demo](samples/tensorflow_performance/README.md)
* [PyTorch demo](samples/pytorch_performance/README.md)
* [TensorFlow demo, no model modifier required](samples/tensorflow_no_model_modifier_performance/README.md)

## License

BERT model optimization is licensed under [Apache License Version 2.0](LICENSE). Refer to the
"[LICENSE](LICENSE)" file for the full license text and copyright notice.

This distribution includes third party software governed by separate license
terms.

Apache License Version 2.0:

* [Google AI BERT](https://github.com/google-research/bert)
* [Tensorflow tutorials](https://github.com/tensorflow/text/tree/master/docs/tutorials)
* [Intel® oneAPI Deep Neural Network Library (oneDNN)](https://github.com/oneapi-src/oneDNN)
* [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models)

## Features and known issues

See [ChangeLog](CHANGELOG.md)

## Support

Please submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/intel/light-model-transformer/issues) page.
