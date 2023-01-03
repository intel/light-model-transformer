# Prepare Hugging Face model for modification
Hugging Face models for Tensorflow are available in h5 format. In order to be able to modify it, we have to change format to saved model format.
> This guide was tested with bert-base-cased, bert-base-uncased, bert-large-cased and bert-large uncased.
## Downloading the model
Hugging Face models are available to download from their website  
`https://huggingface.co/models`   
You can download them using commands below:
```sh
git lfs install
git clone https://huggingface.co/<hugging-face-model-name>
```

Alternatively you can download the model by using their `transformers` API:
```python
from transformers import BertTokenizer, TFBertForSequenceClassification
tokenizer = BertTokenizer.from_pretrained(<hugging-face-model-name>)
model = TFBertForSequenceClassification.from_pretrained(<hugging-face-model-name>)
```

## Fine tuning the model with MRPC dataset
We recommend using [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/tensorflow/text-classification/run_glue.py) script from Hugging Face for train (and download). You can run it with a path to already downloaded model or use model identifier to download model from Hugging Face. Than pick the name of glue dataset that you want to do fine tune on. In our case it was MRPC. Please remember to add the `--do_train` flag. We also changing the default learning rate from 5e-05 to 2e-05 by using parameter `--learning_rate` and number of epochs from default 3 to 10 with parameter `--num_train_epochs`. We are also adding `--do_eval` to check the accuracy at the end of a training.

```sh
$ python run_glue.py \
    --model_name_or_path <hugging-face-model-name-or-path> \
    --task_name mrpc \
    --do_train \
    --do_eval \
    --learning_rate 2e-05 \
    --num_train_epochs 10 \
    --output_dir /tmp/output/
```
## Converting model to saved model
Since our modification scripts are working on saved model format, we will now convert Hugging Face model in h5 format to saved model format.

```python
        from transformers import TFBertForSequenceClassification
        import tensorflow as tf
        model = TFBertForSequenceClassification.from_pretrained(<hugging-face-model-name>)
        callable = tf.function(model.call)
        concrete_function = callable.get_concrete_function([
            tf.TensorSpec([None, None], tf.int32, name="attention_mask"),
            tf.TensorSpec([None, None], tf.int32, name="input_ids"),
            tf.TensorSpec([None, None], tf.int32, name="token_type_ids")])
        tf.saved_model.save(model, "/tmp/saved_model", signatures=concrete_function)
```

After this step you will have Hugging Face model in saved model format in location `/tmp/saved_model/`.
> Please note that we use the `TFBertForSequenceClassification` to preserve the classification head graph in model because other class may cut this part of model out.