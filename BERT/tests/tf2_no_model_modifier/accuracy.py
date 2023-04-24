#!/usr/bin/env python

import numpy as np
import pandas as pd

import time
import transformers

from datasets import load_dataset

import itertools
import argparse

def run_model(model, config, inputs, labels, results):
    try:
        model = transformers.TFBertForSequenceClassification.from_pretrained(model, config=config)
    except OSError as e:
        # OSError is thrown when there is no .h5 weights file for the TensorFlow model.
        # If PyTorch is istalled, we can try to load from a torch weights.
        # This works e.g. for the Intel/bert-base-uncased-mrpc.
        # If this fails, or if torch is not available, we are out of options. 
        if transformers.is_torch_available():
            model = transformers.TFBertForSequenceClassification.from_pretrained(model, config=config, from_pt=True)
        else:
            raise RuntimeError('Could not load the TensorFlow model directly. Cannot attempt loading the model '
                               'from PyTorch weights, becasue torch is not installed.')

    start = time.perf_counter()
    output = model(**inputs).logits
    end = time.perf_counter()
    output = np.argmax(output.numpy(), -1)

    correct = np.sum(labels == output)
    num_samples = len(labels)
    latency = end - start
    throughput = num_samples / latency
    return correct, num_samples, 1000 * latency, throughput
    

def main(args: argparse.Namespace):
    dataset = load_dataset('glue', 'mrpc')
    data = dataset['test']

    samples = data[:args.num_samples]
    labels = np.array(samples['label'])

    tokenizer = transformers.BertTokenizer.from_pretrained(args.tokenizer)
    inputs = tokenizer(text=samples['sentence1'], text_pair=samples['sentence2'], return_tensors='tf', max_length=512, padding='max_length')

    results = pd.DataFrame(columns=['Model', 'Quantization', 'BFloat16', 'Correct/Total', 'Accuracy', 'Latency [ms]', 'Throughput [samples/s]'])


    # Unoptimized model run
    print('Running unoptimized model')
    config = transformers.BertConfig.from_pretrained(args.model)
    correct, num_samples, latency, throughput = run_model(args.model, config, inputs, labels, results)
    row = pd.Series({
        'Model': args.model,
        'Quantization': '-',
        'BFloat16': '-',
        'Correct/Total': f'{correct}/{num_samples}',
        'Accuracy': f'{correct / num_samples}',
        'Latency [ms]': f'{latency}',
        'Throughput [samples/s]': f'{throughput}',
        })
    results = pd.concat([results, row.to_frame().T], ignore_index=True)

    # Optimized model runs:
    import bert_op

    quant_values = [False, True]
    bf_values = [False]

    for quant, bf16 in itertools.product(quant_values, bf_values):
        config = transformers.BertConfig.from_pretrained(args.model)
        print(f'Running optimized model: Quantization = {quant}, BFloat16 = {bf16}')
        config.use_quantization = quant
        config.use_bfloat16 = bf16
        if quant:
            config.quant_factors_path = args.quant_factors_path
        correct, num_samples, latency, throughput = run_model(args.model, config, inputs, labels, results)
        row = pd.Series({
            'Model': '(OPT) ' + args.model,
            'Quantization': 'On' if quant else 'Off',
            'BFloat16': 'On' if bf16 else 'Off',
            'Correct/Total': f'{correct}/{num_samples}',
            'Accuracy': f'{correct / num_samples}',
            'Latency [ms]': f'{latency}',
            'Throughput [samples/s]': f'{throughput}',
            })
        results = pd.concat([results, row.to_frame().T], ignore_index=True)

    print(results.to_markdown())

    if args.csv is not None:
        results.to_csv(args.csv, sep='\t', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='An MRPC accuracy script for HuggingFace BERT models optimized with the BertOp.')

    parser.add_argument('-m', '--model', type=str, default='Intel/bert-base-uncased-mrpc')
    parser.add_argument('-t', '--tokenizer', type=str, default='bert-base-uncased', help='Tokenizer to used for the model.')

    parser.add_argument('-p', '--quant-factors-path', type=str, default='', help='Path to the quantization factors file.')

    parser.add_argument('-n', '--num_samples', type=int, default='100', help='Number of samples to run.')

    parser.add_argument('--output-csv', dest='csv', type=str, default=None,
                        help='If provided, results will be saved to a .csv file at this location.')
    parser.add_argument('--sep', default='\t', help='Separator used when writing to the .csv file.')


    args = parser.parse_args()
    main(args)
