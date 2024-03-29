# BERT Model Optimization Change Log

## v0.10 - 2023-04

### Added

* Implemented approach to optimize HuggingFace BERT in TensorFlow without usage of [model_modifier](python/model_modifier/).
  See the appropriate [sample](samples/tensorflow_no_model_modifier_performance/)
* Changed oneDNN API version support from 2.x to 3.x - oneDNN version <3.0 is not supported anymore

## v0.9 - 2023-04

### Added

* Completed PyTorch integration. See [tests README](tests/pytorch/README.md) for details
* TF and PT integration demos. See [samples](samples/) directory
* Performance optimizations including:
  * Use oneDNN Convolution 1x1 for linear operations
  * Optimized infrastructure code
* Added patterns to modify RoBERTa models. See pattern files [here](util/tf2-hf-roberta/)

## v0.8 - 2023-02-22

### Added

* Fixed INT8 accuracy issue for sequence len > 128
* BF16 optimization support - more than 2.5x performance gain on 4th Gen Intel® Xeon® Scalable processors
* Huggingface BERT models support (see the [README](util/tf2-hf/README.md))
* Experimental PyTorch integration (use cmake -DBACKENDS=PT to enable)

### Known Issues

* PyTorch integration performance

## v0.7 - 2022-11-02

### Added

* Variadic BERT parameters: sequence length, hidden size, intermediate size, layers number
* BERT-Large model support
* INT8 quantization factors calibration mode - see [Calibration mode](python/README.md#calibration-mode) section in [Python tools documentation](python/README.md)

### Known Issues

* INT8 quantization mode has an accuracy issue for sequence length > 128
* Maximum supported TensorFlow version is 2.9

## v0.5 - 2022-06-28

### Added

* Initial release
* Enabled BERT Base with fixed sequence length 128
* Added support for TensorFlow v.1.15, v.2.x

### Known Issues

* BERT model variants support limited to BERT-Base SeqLen=128
