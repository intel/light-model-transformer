# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import time

import numpy as np

import tensorflow as tf
import tensorflow_text # Dependency of tf
from datasets import load_dataset
from transformers import BertTokenizer

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='GLUE/MRPC accuracy testing script for BERT models with monolithic BertOp.')

    parser.add_argument('model_dir', metavar='model-dir',
                        type=str, help='Path to the BERT model.')
    parser.add_argument('op_library', metavar='op-library',
                        type=str, help='Path to the .so containing the BertOp.')
    parser.add_argument('-g', '--hugging_face', type=str, default=None, 
                        help='Name of Hugging Face model.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true',
                        help='Print progress while running and a list of every label-prediction pair.')
    parser.add_argument('--out-file', type=str, help='Path to the output .csv file.')

    args = parser.parse_args()

    print('### Loading op library.')

    tf.load_op_library(args.op_library)

    print('### Preparing dataset.')

    dataset = load_dataset("glue","mrpc")

    validation_dataset = dataset['validation']
    
    sentence1_key = "sentence1"
    sentence2_key = "sentence2"
    
    total_samples = len(validation_dataset['label'])
    labels = validation_dataset['label'][:total_samples]
    
    if args.hugging_face is not None:
        tokenizer = BertTokenizer.from_pretrained(args.hugging_face)
        
        def preprocess_function(examples):
            args = (examples[sentence1_key], examples[sentence2_key])
            result = tokenizer(*args, padding="max_length", max_length=128, return_tensors='tf')
            return result
       
        validation_dataset = validation_dataset.map(preprocess_function, batched=True)
        val_dict = validation_dataset.to_dict()
        
        for key in ["attention_mask", "input_ids", "token_type_ids"]:
            val_dict[key] = tf.constant(val_dict[key], dtype=tf.int32)

    print('### Loading the model.')

    model = tf.saved_model.load(args.model_dir)

    print('### Testing the model.')

    start = time.time()

    if args.hugging_face is not None:
        res = model.signatures["serving_default"](attention_mask=val_dict["attention_mask"], input_ids=val_dict["input_ids"], token_type_ids=val_dict["token_type_ids"])
        res = res["logits"].numpy()
    else:
        res = model([
            validation_dataset[sentence1_key][:total_samples],
            validation_dataset[sentence2_key][:total_samples]
        ])
        res = res.numpy()
    
    end = time.time()

    results = np.argmax(res, axis=1)
        
    if args.verbose:
        print(f"{'Label' : <10} | {'Response' : <10}")
        for i in range(total_samples):
            print(f'{labels[i] : <10} | {results[i] : <10}  {res[i]}')

    correct = np.sum(results == labels)
    accuracy = correct / total_samples
    print(f'Accuracy: {correct} / {total_samples} - {accuracy * 100}%')
    print(f'Elapsed time: {end - start}')
    
    if args.out_file is not None:
        with open(args.out_file, 'a') as f:
            f.write(f'{correct}/{total_samples}\t{correct/total_samples}\n')
