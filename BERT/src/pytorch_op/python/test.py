#!/usr/bin/env python

import torch
import torch.utils.benchmark as benchmark
# import intel_extension_for_pytorch
import numpy as np
import timeit

import transformers
from datasets import load_dataset

import os

def run_benchmark(model, data, batch_size, tokenizer, num_runs=1, **kwargs):
    t = benchmark.Timer(
        stmt="""\
            with torch.no_grad():
                model(**inputs)
            """,
        setup="""\
            batch = data.shuffle()[:batch_size]
            inputs = tokenizer(text=batch['sentence1'], text_pair=batch['sentence2'], return_tensors='pt', max_length=512, padding='max_length')
            """,
        globals={
            'model': model,
            'data': data,
            'batch_size': batch_size,
            'tokenizer': tokenizer
            },
        num_threads=os.cpu_count() // 2,
        **kwargs
    )
    return t.timeit(num_runs)
    # return t.blocked_autorange(min_run_time=60)

dataset = load_dataset('glue', 'mrpc')
data = dataset['test']
batch_size = 32
num_runs = 2

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

model = transformers.BertForSequenceClassification.from_pretrained("Intel/bert-base-uncased-mrpc")
model.eval()

results = []

result = run_benchmark(model, data, batch_size, tokenizer, num_runs,
    label='BERT-Base',
    sub_label=f'Batch: {batch_size}',
    description='Vanilla PyTorch, Original HF Model')
results.append(result)

import bert_op
model = transformers.BertForSequenceClassification.from_pretrained("Intel/bert-base-uncased-mrpc")
model.eval()


result = run_benchmark(model, data, batch_size, tokenizer, num_runs,
    label='BERT-Base',
    sub_label=f'Batch: {batch_size}',
    description='Vanilla PyTorch, BertEncoderOp Model')

results.append(result)

compare = benchmark.Compare(results)
compare.print()
