# Testing custom TF op for bert model

## Introduction

Quick tutorial on how to test accuracy of a model modified to use custom bert operator

## Requirements

- https://github.com/google-research/bert  
- run_classifier.py from "put path to rafal branch when its merged"  
- Custom TF operator for bert model  
- Dataset to test it on (instructions on how download MRPC, provided in [here](https://github.com/intel-sandbox/cesg.bert.utils/blob/master/users/rbogdano/scripts/run_classifier_guide.txt) )  
- Google bert model (english or chinese version, can be download from https://github.com/google-research/bert)  
- Python 3.5–3.8 (Required for tensorflow 1.15)
- tensorflow 1.15
- pandas

## Usage 

1. Run run_classifier.py to fine tune and create a saved model pb file.  (instructions in [here](https://github.com/intel-sandbox/cesg.bert.utils/tree/master/users/rbogdano/scripts))
2. Run main.py to create modified model with 1 bert node at path_to_modified_model

```sh
export PYTHONPATH=$PYTHONPATH:<path_to_bert_repo>
path_to_model=<path to original saved_model>
path_to_bertop=<path to optimized BERT op>
path_to_modified_model=<output path to put the modified model in>

python main.py $path_to_model $path_to_bertop $path_to_modified_model
```

4. Run accuracy.py to get accuracy results

```sh
data_dir=<dataset dir> # same dataset that was used to finetune the model in run_classifier)
vocab_path=<path to vocabtxt> # this file will be located in same folder as google bert model files, for example in case of english model it will be in folder uncased_L-12_H-768_A-12)

python accuracy.py $path_to_modified_model $data_dir $path_to_bertop --vocab_file=$vocab_path
```

