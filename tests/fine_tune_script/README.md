# Fine Tune script for BERT model

This is fine-tune script which must be run on already pre-trained model. This action is performed to enhance accuracy of the model. This script will generate saved_model in the location where it was runned.

Original script is available
https://github.com/google-research/bert/blob/master/run_classifier.py

## Prerequisites

 1. Install TensorFlow 1.15
 2. Download General Language Understanding Ealuation (GLUE) data by running this script with below command https://gist.github.com/vlasenkoalexey/fef1601580f269eca73bf26a198595f3
    ```bash
    $ python3 download_glue_data_script.py --data_dir glue_data --tasks MRPC
    ```
 3. In the same folder where run_classifier.py script is you have to have below files. These files are available to download from google bert repository (https://github.com/google-research/bert):
    - modeling.py
    - optimization.py
    - tokenization.py
 4. Setup environmental variables
    - GLUE_DIR - place where should be folder named MRPC with data 
    - BERT_BASE_DIR - directory to ml model (example: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

## Example

```bash
$ export GLUE_DIR=.
$ export BERT_BASE_DIR=/home/rav/Documents/uncased_L-12_H-768_A-12

$ python3 run_classifier.py --task_name=MRPC --do_train=true --do_eval=true --do_export=true --do_predict=true  --data_dir=$GLUE_DIR/MRPC   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --max_seq_length=128   --train_batch_size=32   --learning_rate=2e-5   --num_train_epochs=3.0   --output_dir=/tmp/mrpc_output/
```



