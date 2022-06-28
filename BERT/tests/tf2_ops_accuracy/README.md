# Testing custom TF op for bert model, tf2 version

## Introduction

Quick tutorial on how to test accuracy of a model with a custom bert operator using TensorFlow 2.x


## Requirements 

See [requirements.txt](../../requirements.txt).


## Fine tuning a model

Fine tuning a BERT model is done using the `run_classifier.py` tool.

### Usage of `run_classifier.py`
```sh
python run_classifier.py -b $encoder_handle -p $preprocessor_handle -e $epochs -l $learning_rate $output_dir 
```

The tool will build a model consisting of the specified preprocessor and bert encoder, add a classification head to it
and fine-tune it using the GLUE/MRPC dataset.

### Default model

* Run the `run_classifier.py` tool.

All arguments except `output_dir` have defaults and are optional. The [default BERT model][default-bert] will be
downloaded, fine-tuned and saved in `output_dir`.

### Select a different model - [model modification](#modify-the-model) may not work

* Pick a BERT encoder from here: https://tfhub.dev/google/collections/bert/1.
* Find a compatible preprocessor model for your bert encoder here:
https://www.tensorflow.org/text/tutorials/classify_text_with_bert#choose_a_bert_model_to_fine-tune.
* (Optional) Download the models.
* Run the `run_classifier.py` tool. All arguments except `output_dir` have defaults and are optional, but you want to
use your selected models, so you must provide handles for TF Hub:
    * If you downloaded the models, the handles are the download paths.
    * Otherwise, provide the TF Hub URL. (TF Hub will cache the downloads between runs.)


## Modify the model

Modifying the model to use the monolithic BertOp is done using the [model modifier tool][model-modifier].
Rather than calling the tool directly, it's easier to use a script preconfigured for a specific BERT model. Those
scripts are available in the [util directory][util-dir]. Currently only the [default model][default-bert] is
supported.

The steps to modify the model are:

* (If not done previously) Compile the model modifier protos:
```sh
$ cd <repo_root>/python/proto
$ ./compile_proto.sh $tensorflow_include_dir
```
For TF2, `tensorflow_include_dir` will be `<...>/site-packages/tensorflow/include`. \
TF1 does not seem to include `.proto` files with the package. In this case, Tensorflow sources can be used.
`tensorflow_include_dir` will then be the root of the Tensorflow repo.

* Put the model modifier on `PYTHONPATH`:
```sh
$ export PYTHONPATH=<repo_root/python>:$PYTHONPATH
```

* Run the preconfigured script:
```sh
$ cd <repo_root>/util/tf2/bert_en_uncased_L-12_H-768_A-12_4
$ ./replace_full_bert $path_to_bert_encoder $path_to_fine_tuned_model
```
The `path_to_bert_encoder` should be the BERT encoder model, as downloaded from TF Hub. \
The `path_to_fine_tuned_model` is the path to the output model of `run_classifier.py` \
The preconfigured script will use the model modifier tools to generate a BERT `pattern.pb` from the original encoder
model, combine it with the `fused_bert_node_def.pb` included with the script to create a `recipe.pb`, then use that
recipe to locate the BERT pattern in the fine-tuned model and replace it with the fused BERT op. The intermediate
`pattern.pb` and `recipe.pb` are created in a temporary directory and removed when finished.

This requires the original BERT encoder model to be available, which is not ideal and may be changed in the future.
For example, the `recipe.pb` can be stored in the repository.

* Enable the modified model graph.

The tool above creates a `modified_saved_model.pb` next to the `saved_model.pb` of the fine-tuned model. To use the
modified graph, do:
```sh
$ cd $path_to_fine_tuned_model
$ mv saved_model.pb original_saved_model.pb
$ ln -s modified_saved_model.pb saved_model.pb
```
This will preserve the original model graph, but calls to `tf.saved_model.load(...)` will not use the modified version.

## Bert op configuration

There is also a need to set bert op configuration, full tutorial on all possible options can be found [here](../../python/README.md)  
However to launch this particular model with no quanitization and no bfloat16 you can do the following
```
export PYTHONPATH=$PYTHONPATH:$<repo_root>/python
python -m model_modifier.configure_bert_op --no-quantization --no-bfloat16 $path_to_fine_tuned_model
```


## Accuracy checking

Accuracy checking can be done using the `accuracy.py` script.

```sh
python accuracy.py $model_dir $op_library
```
The `model_dir` argument is the path to the modified BERT model. \
The `op_library` argument is the path to the .so containing the BertOp definition. 

The script should also work on unmodified models, which can be used as a baseline to measure accuracy loss of the
BertOp.

[default-bert]: https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4
[model-modifier]: ../../python/model_modifier
[util-dir]: ../../util
