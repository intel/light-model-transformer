# Testing custom TF op for bert model

## Introduction

Quick tutorial on how to test accuracy of a model modified to use custom bert operator

## Requirements

- https://github.com/google-research/bert  
- Custom TF operator for bert model  
- Dataset to test it on (instructions on how download MRPC, provided below)
- Google bert-base model (https://github.com/google-research/bert)  
- Python 3.5–3.8 (Required for tensorflow 1.15)
- tensorflow 1.15
- pandas

## Fine tuning a model

To download MRPC and fine tune bert-base model, follow the tutorial in [fine tune script](../fine_tune_script/README.md)

## Modify the model

Modifying the model to use the monolithic BertOp is done using the [model modifier tool][model-modifier].
Rather than calling the tool directly, it's easier to use a script preconfigured for a specific BERT model. Those
scripts are available in the [util directory][util-dir].

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
$ cd <repo_root>/util/tf1/bert_en_uncased_L-12_H-768_A-12
$ ./replace_full_bert $path_to_model $path_to_model
```
The `path_to_model` is the path to the bert model which is an output of the run classiflier script.
The preconfigured script will use the model modifier tools to generate a BERT `pattern.pb` from the original encoder
model, combine it with the `fused_bert_node_def.pb` included with the script to create a `recipe.pb`, then use that
recipe to locate the BERT pattern in the fine-tuned model and replace it with the fused BERT op. The intermediate
`pattern.pb` and `recipe.pb` are created in a temporary directory and removed when finished.

* Enable the modified model graph.

The tool above creates a `modified_saved_model.pb` next to the `saved_model.pb` of the fine-tuned model. To use the
modified graph, do:
```sh
$ cd $path_to_model
$ mv saved_model.pb original_saved_model.pb
$ ln -s modified_saved_model.pb saved_model.pb
```
This will preserve the original model graph, but calls to `tf.saved_model.load(...)` will not use the modified version.

## Bert op configuration

There is also a need to set bert op configuration, full tutorial on all possible options can be found [here](../../python/README.md)  
However to launch this particular model with no quanitization and no bfloat16 you can do the following
```
export PYTHONPATH=$PYTHONPATH:$<repo_root>/python
python -m model_modifier.configure_bert_op --no-quantization --no-bfloat16 $path_to_model
```

## Accuracy checking

```sh
data_dir=<dataset dir> # same dataset that was used to finetune the model in run_classifier
vocab_path=<path to vocabtxt> # this file will be located in same folder as google bert model files, for example in case of english model it will be in folder uncased_L-12_H-768_A-12)

python accuracy.py --model_path=$path_to_model --data_dir=$data_dir --op_path=$path_to_bertop --vocab_file=$vocab_path 
```
[model-modifier]: ../../python/model_modifier
[util-dir]: ../../util
