BERT model optimization
=======================

Bert model optimization is an open-source optimization for bert language processing model (only BERT-base model supported at this moment).  
The optimization is based on [Bfloat16 Optimization Boosts Alibaba Cloud BERT Model Performance](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/alibaba-blog.html).  
Furthermore, it utilizes  [Intel® oneAPI Deep Neural Network Library (oneDNN)](https://github.com/oneapi-src/oneDNN) to obtain additional performance gains.  
Bert model optimization is split into two parts, model modifier, which modifies the model to use a custom operator and a custom operator which utilizes oneDNN.  
Currently only models built using tensorflow 1.x and 2.x are supported.

# Table of contents

- [Building from source](#Building-from-source)
- [Requirements for building from source](#Requirements-for-building-from-source)
- [Getting started](#Getting-started)
- [License](#license)
- [Security](#security)
- [Support](#support)


# Building from source 
1. Clone and build:
```sh
git clone ...
cd libraries.ai.performance.models.bert
mkdir build
cd build
source /opt/intel/oneapi/setvars.sh # Make sure CMake can find oneDNN
cmake ..
cmake --build . -j 8
```
2. Run benchmark: `tests/benchmark/benchmark`


# Requirements for building from source
Bert model optimization supports systems meeting the following requirements:
* [oneDNN Library](https://github.com/oneapi-src/oneDNN) 2.6 or later
* C++ compiler
* [CMake](https://cmake.org/download/)
* Linux based operating system 

# Getting started 
For the currently supported use cases, short tutorials on usage are provided.
All of them require built from source bert operator, refer to [Building from source](#Building-from-source)
* [tensorflow 1.x](tests/tf1_ops_accuracy/README.md)
* [tensorflow 2.x](tests/tf2_ops_accuracy/README.md)
* [Model Zoo for Intel® Architecture](tests/model_zoo/README.md)
# License 

Bert model optimization is licensed under [Apache License Version 2.0](LICENSE). Refer to the
"[LICENSE](LICENSE)" file for the full license text and copyright notice.

This distribution includes third party software governed by separate license
terms.

Apache License Version 2.0:
* [Google AI bert](https://github.com/google-research/bert)
* [Tensorflow tutorials](https://github.com/tensorflow/text/tree/master/docs/tutorials)
* [Intel® oneAPI Deep Neural Network Library (oneDNN)](https://github.com/oneapi-src/oneDNN)
* [Model Zoo for Intel® Architecture](https://github.com/IntelAI/models)

# Support 

Please submit your questions, feature requests, and bug reports on the [GitHub issues](https://github.com/intel/light-model-transformer/issues) page.

