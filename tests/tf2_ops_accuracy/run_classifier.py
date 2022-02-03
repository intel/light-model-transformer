# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow_text as text  # must be imported to load preprocessor ops
from official.nlp import optimization  # to create AdamW optimizer

import sys
import argparse

tfhub_handle_preprocess_default = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder_default = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
epochs_default = 5
init_lr_default = 2e-5


def build_classifier_model() -> tf.keras.Model:
    preprocessor = hub.load(
        tfhub_handle_preprocess)

    text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentence1'),
                   tf.keras.layers.Input(shape=(), dtype=tf.string, name='sentence2')]  # This SavedModel accepts up to 2 text inputs.
    tokenize = hub.KerasLayer(preprocessor.tokenize)
    tokenized_inputs = [tokenize(segment) for segment in text_inputs]

    seq_length = 128  # maxTokenSize of BertOp
    bert_pack_inputs = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length))  # Optional argument.
    encoder_inputs = bert_pack_inputs(tokenized_inputs)

    encoder = hub.KerasLayer(tfhub_handle_encoder,
                             trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(2, activation=None, name='classifier')(net)
    net = tf.sigmoid(net)
    return tf.keras.Model(text_inputs, net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune a BERT model for accuracy testing using the GLUE dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('output_dir', metavar='output-dir',
                        help='Directory to save the tuned model in.')
    parser.add_argument('-b', '--bert', default=tfhub_handle_encoder_default,
                        help=f'TF Hub handle for the encoder model. This can be a local directory or a URL.')
    parser.add_argument('-p', '--preprocessor', default=tfhub_handle_preprocess_default,
                        help=f'TF Hub handle for the preprocessor model. This can be a local directory or a URL.')
    parser.add_argument('-e', '--epochs', default=epochs_default, type=int,
                        help=f'How many training epochs to run.')
    parser.add_argument('-l', '--learning-rate', default=init_lr_default, type=float,
                        help=f'Initial learning rate, typically in range [2e-5:5e-5].')

    args = parser.parse_args()

    tfhub_handle_preprocess = args.preprocessor
    tfhub_handle_encoder = args.bert

    print('### Preparing dataset.')

    dataset, info = tfds.load('glue/mrpc', with_info=True,
                              # It's small, load the whole dataset
                              batch_size=-1)

    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
    test_dataset = dataset['test']

    print('### Preparing losses and metrics.')

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(
        'accuracy', dtype=tf.float32)]

    print('### Preparing training parameters and optimizer.')

    epochs = args.epochs
    steps_per_epoch = len(train_dataset['label'])
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)
    init_lr = args.learning_rate
    print(f'epochs: {epochs}')
    print(f'steps_per_epoch: {steps_per_epoch}')
    print(f'num_train_steps: {num_train_steps}')
    print(f'num_warmup_steps: {num_warmup_steps}')
    print(f'init_lr: {init_lr}')

    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    print(f'### Preparing the model: {tfhub_handle_encoder}.')

    classifier_model = build_classifier_model()

    print('### Compiling the model.')
    classifier_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics)

    print('### Training the model.')
    history = classifier_model.fit(x=train_dataset,
                                   y=train_dataset['label'],
                                   validation_data=(
                                       validation_dataset, validation_dataset['label']),
                                   epochs=epochs)

    print('### Evaluating the model.')

    loss, accuracy = classifier_model.evaluate(
        x=validation_dataset, y=validation_dataset['label'])

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    classifier_model.save(args.output_dir)
