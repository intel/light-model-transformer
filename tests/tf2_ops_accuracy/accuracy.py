# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text  # must be imported to load preprocessor ops

import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='GLUE/MRPC accuracy testing script for BERT models with monolithic BertOp.')

    parser.add_argument('model_dir', metavar='model-dir',
                        type=str, help='Path to the BERT model.')
    parser.add_argument('op_library', metavar='op-library',
                        type=str, help='Path to the .so containing the BertOp.')

    args = parser.parse_args()

    print('### Loading op library.')

    tf.load_op_library(args.op_library)

    print('### Preparing dataset.')

    dataset, info = tfds.load('glue/mrpc', with_info=True,
                              # It's small, load the whole dataset
                              batch_size=-1)

    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    print('### Loading the model.')

    model = tf.saved_model.load(args.model_dir)

    print('### Testing the model.')

    count = len(validation_dataset['label'])
    # count = 10

    labels = validation_dataset['label'][:count]

    # Force batch size 1
    results = []
    for i in range(count):
        res = model([
            validation_dataset['sentence1'][i:i+1],
            validation_dataset['sentence2'][i:i+1]
        ])
        res = np.argmax(res.numpy())
        results.append(res)

        s = f'Testing: {i :>3} / {count : <3}'
        print(f"{s}{' ' * (os.get_terminal_size().columns - len(s))}",
              end='\r', flush=True)

    print(f"{'Label' : <10} | {'Response' : <10}")
    for i in range(count):
        print(f'{labels[i] : <10} | {results[i] : <10}')

    correct = np.sum(results == labels)
    accuracy = correct / count
    print(f'Accuracy: {correct} / {count} - {accuracy * 100}%')
