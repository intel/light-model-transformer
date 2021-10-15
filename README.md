# BERT model optimization

## Introduction

Implementation is based on the algorithm described in the article: [Bfloat16 Optimization Boosts Alibaba Cloud BERT Model Performance](https://www.intel.com/content/www/us/en/artificial-intelligence/posts/alibaba-blog.html). Onlpementation utilizes [oneDNN Library](https://github.com/oneapi-src/oneDNN).

## Building from source

1. Install [oneDNN Library](https://github.com/oneapi-src/oneDNN)
2. Clone and build:
```sh
git clone ...
cd libraries.ai.performance.models.bert
mkdir build
cd build
cmake -Ddnnl_DIR=<oneDNN-install-prefix>/lib/cmake/dnnl ..
cmake --build . -j 8
```
3. Run benchmark: `tests/benchmark/benchmark`

## Notes

@rfsaliev: Current sources are based on int8 code from [libraries.ai.bert](https://github.com/intel-sandbox/libraries.ai.bert)
