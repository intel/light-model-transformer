# Testing the custom BERT operator for PyTorch

## Introduction

This is a tutorial on now to optimize huggingface BERT models using the custom operator.

## Requirements

- the operator built with `-DBACKENDS=PT`
- python dependencies, see [here](../../requirements-pt.txt) (IPEX is optional)
- Make sure the path to the compiled operator .so is exported in the `BERT_OP_PT_LIB` environment variable:
```sh
export BERT_OP_PT_LIB=/<path-to-build-dir>/src/pytorch_op/libBertOpPT.so
```
- Add the `python` subdirectory of this project to you PYTHONPATH:
```sh
export PYTHONPATH=$PYTHONPATH:<repo-root>/python
```

## Accuracy demo

Navigate to the `tests/pytorch` subdirectory of the project and run the accuracy script:
```sh
cd <repo-root>/tests/pytorch
python accuracy.py
```
This will execute a default MRPC accuracy check with the following configuration:
* [Intel/bert-base-uncased-mrpc](https://huggingface.co/Intel/bert-base-uncased-mrpc) model
* [bert-base-uncased](https://huggingface.co/bert-base-uncased) tokenizer
* 100 samples

You can run `python accuracy.py -h` to view available options.

The accuracy script will first execute the huggingface model as-is. Then, it will be optimized with the BERT operator,
and the same test samples will be fed to the model in the following modes:
* pure FP32
* FP32 + QINT8
* BF16
* BF16 + QINT8

A summary of the accuracy scores will then be printed to the console.

## Performance demo

Navigate to the `tests/pytorch` subdirectory of the project and run the benchmark script with the desired configuration,
for example:

```sh
cd <repo-root>/tests/pytorch
python benchmark.py -m bert-large-uncased --bert-op --quantization --batch-size=4 --seq-len=128 --run-time=60
```

This will load the `bert-large-uncased` model, optimize it with the BERT operator, then execute in QINT8 mode,
with a batch size of 4 and sequence length of 128. The benchmark will first run a number of warmpu cycles (defaults to
10% of the measured run time, so 6 seconds in this case), then measure the average latency and throughput over 60
seconds.

Run `python benchmark.py -h` for a full lits of options.

For comparison, you can then run the same benchmark without the `--bert-op` flag to execute the unoptimized model. In
both cases, the models are first passed through `torch.jit.trace` and `torch.jit.freeze`.

### Easy sample

The `benchmark.py` script is also used in the docker-based sample found [here](../../samples/pytorch_performance). This is the
easiest way to see the BERT operator in action. refer to the [README](../../samples/pytorch_performance/README.md) for
details.


## Optimizing your own workflow

Using the BERT operator in your model is very easy. Currently, all huggingface BERT models should benefit from this
optimization, i.e. all models that use the `transformers.models.bert.modeling_bert.BertEncoder` class.

In order to start using the optimized BERT op, just import the `bert_op` package in your code, before you load the
model. Assuming you have the environment set up and the operator is compiled (see [Requirements](#requirements)), 
adding the `import bert_op` line should be all you need, for example:

```python

import transformers

... # your code

import bert_op # Important, do this at any point BEFORE the call to `transformers.from_pretrained`
model = transformers.from_pretrained('bert-base-uncased')

output = model(**inputs) # model now executes the BERT operator.

```

That's it! Your model is now using the BERT operator.

### Caveats

1. The operator only supports inference workloads, and currently only works on BERT models. (The Tensorflow operator
has also been tested on RoBERTa models, which will likely be added to the Pytorch operator as well.)

2. The optimiziation is injected into the model via class substitution:
    ```python
    transformers.models.bert.modeling_bert.BertEncoder = BertEncoderOp
    ```
    This means that any model which uses `transformers.models.bert.modeling_bert.BertEncoder`, will use
    `bert_op.BertEncoderOp` instead, if it is created after `import bert_op`. Models created before the import are
    unaffected.


## Additional options

### QINT8 and BF16

The BERT operator can utilize Int8 quantization and BFloat16 computations on supported hardware. To enable these features,
add appropriate fields to the BertConfig before loading the model:

```python
import transformers

... # your code

import bert_op # Important, do this at any point BEFORE the call to `transformers.from_pretrained`
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
config.use_bfloat16 = True # or False, defaults to False if not provided
config.use_quantization = True # or False, defaults to False if not provided
config.quantization_factors = [...] # List of quantization factors, necessary if config.use_quantization == True
                                    # See [below](#quantization-factor-calibration) for instruction on how to get this
model = transformers.BertModel.from_pretrained('bert-base-uncased', config=config)

output = model(**inputs)

```

### Quantization factor calibration

NOTE: Calibration only works in eager mode. Trying to `torch.jit.trace` or `torch.jit.freeze` the model in calibration
mode will fail.

When using the PyTorch backend, the BERT operator accepts quantization factors from a simple python list in the BertConfig.
In order to get these factors, prepare the BertConfig for FP32 mode and set the calibration flag, execute some calibration
workload, then retrieve the factors from the `BertEncoderOp`:

```python
import transformers

... # your code

import bert_op # Important, do this at any point BEFORE the call to `transformers.from_pretrained`
config = transformers.BertConfig.from_pretrained('bert-base-uncased')
config.use_bfloat16 = False     # These two can be omitted, 
config.use_quantization = False # they are here just for clarity
config.calibrate_quant_factors = True # This enables calibration mode
model = transformers.BertModel.from_pretrained('bert-base-uncased', config=config)

# IMPORTANT: no tracing and freezing the model here. The below fill fail in calibration mode:
# model = torch.jit.trace(model, data, strict=False)
# model = torch.jit.freeze(model)

output = model(**inputs) # model now executes in FP32 mode, and `TFBertEncoderOp` calibrates the quantization factors

quant_factors = model.encoder.get_quantization_factors() # Save the contents of quant_factors so that they can be reused
                                                         # in subsequent executions

```

Upon next execution, you can set `config.calibrate_quant_factors = False`, `config.use_quantization = True` and set the
`config.quantization_factors` list to the values previously received from calibration.
