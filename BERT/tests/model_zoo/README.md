# Testing custom tf2 op using Model Zoo for Intel® Architecture
## Introduction

Quick tutorial on how to test accuracy of a model with a custom bert operator using Model Zoo for Intel® Architecture

## Creating frozen graph and modifying it.

You can follow this [tutorial](https://github.com/IntelAI/models/blob/master/benchmarks/language_modeling/tensorflow/bert_base/README.md)
to create and test a bert model. After running  export_classifier, it will give you a frozen graph in OUTPUT_DIR.<br />The steps to modify the model are:

* (If not done previously) Compile the model modifier protos:
```sh
$ cd <repo_root>/python/proto
$ ./compile_proto.sh $tensorflow_include_dir
```
For TF2, `tensorflow_include_dir` will be `<...>/site-packages/tensorflow/include`. 

* Put the model modifier on `PYTHONPATH`:
```sh
$ export PYTHONPATH=<repo_root/python>:$PYTHONPATH
```

* Run the preconfigured script:
```sh
$ cd <repo_root>/util/tf2-github/uncased_L-12_H-768_A-12/
$ ./replace_full_bert.sh $path_to_frozen_graph $path_to_frozen_graph
```
path_to_frozen_graph=$OUTPUT_DIR/frozen_graph.pb if you followed the tutorial<br />
This will result in modified_saved_model.pb (frozen graph in reality) being created in /util/tf2-github/uncased_L-12_H-768_A-12/ which is a modified frozen graph that uses a custom operator.
Rename it to frozen_graph.pb
```
mv modified_saved_model.pb frozen_graph.pb
```

## Bert op configuration

There is also a need to set bert op configuration, full tutorial on all possible options can be read [here](../../python)<br />
However to launch this particular model you can do the following
```
export PYTHONPATH=$PYTHONPATH:$<repo_root>/python
python -m model_modifier.configure_bert_op --no-quantization --no-bfloat16 <path to modified frozen graph created above>
```
by default the path should be <root_repo>/util/tf2-github/uncased_L-12_H-768_A-12/frozen_graph.pb

To test it you have 2 options.

## Inference way 1 (Modify code)
This requires you to modify Model Zoo for Intel® Architecture files by adding 1 line which will load the model.<br />
In both cases for bert large and bert base Model Zoo for Intel® Architecture will run models/models/language_modeling/tensorflow/bert_large/training/fp32/run_classifier.py<br />
The the following line has to be added to the run_classifier.py file at any point before the model is loaded, for example at the end of the file before tf.compat.v1.app.run() is called 
``` 
tf.load_op_library("<repo_root>/build/src/tf_op/libBertOp.so")
``` 
Make sure to set your FROZEN_DIR argument to modified model path instead of non modified model path.<br />
After doing so launch inference using the last code block in this [tutorial](https://github.com/IntelAI/models/blob/master/benchmarks/language_modeling/tensorflow/bert_base/README.md)

## Inference way 2 (No modification)    
If you don't want to modify the code in Model Zoo for Intel® Architecture, use the script in this directory which will launch run_classifier after loading the model.
```
export PYTHONPATH=$PYTHONPATH:$<repo_root>/model_zoo/models/language_modeling/tensorflow/bert_large/inference
FROZEN_DIR=<path_to_modified_frozen_graph>
path_to_bertop=<repo_root>/build/src/tf_op/libBertOp.so
python run_model_zoo.py --frozen_graph_path=$FROZEN_DIR/frozen_graph.pb --output_dir=$OUTPUT_DIR --bert_config_file=$BERT_BASE_DIR/bert_config.json --do_train=False --precision=fp32 --do_lower_case=True --task_name=MRPC --do_eval=true --vocab_file=$BERT_BASE_DIR/vocab.txt --data_dir=$GLUE_DIR/MRPC  --eval_batch_size=1 --experimental_gelu=False  --max_seq_length=128 --inter_op_parallelism_threads=1 --intra_op_parallelism_threads=18 $path_to_bertop
``` 
