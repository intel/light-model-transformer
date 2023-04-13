#!/usr/bin/env python

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import NonCallableMagicMock
import torch
import numpy as np

import transformers

import pandas as pd
import numpy as np

import argparse
import itertools
import random
import time

import logging
import os
from math import ceil

from typing import Optional

LOGGER_FORMAT = '[%(levelname)s][%(name)s]: %(message)s'
logging.basicConfig(format=LOGGER_FORMAT)
log = logging.getLogger(f'{os.path.basename(__file__)}')


def run_benchmark(model: transformers.BertModel, data: torch.Tensor,
                  benchmark_time_seconds: Optional[float] = None, warmup_time_seconds: Optional[float] = None,
                  benchmark_iterations: Optional[int] = None, warmup_iterations: Optional[int] = None):
    """
    We can use seconds rather than number of runs because larger models (and larger batch sizes) typically exhibit
    smaller performance variance between runs, so we need fewer iterations for larger models to get a reasonable
    average throughput. Running the benchmark for a specified time rather than number of iterations accommodates that.
    
    If preferable, it possible to specify the number of iterations instead.
    """

    def timed_run(model: transformers.BertModel, data: torch.Tensor):
        with torch.no_grad():
            start = time.perf_counter()
            model(data)
            end = time.perf_counter()

        return end - start

    def run_for_seconds(model: transformers.BertModel, data: torch.Tensor, run_time_seconds: float):
        elapsed_time_seconds = 0.
        num_runs = 0
        while elapsed_time_seconds < run_time_seconds:
            log.debug(f'Running iteration {num_runs}, elapsed time {elapsed_time_seconds}')
            elapsed_time_seconds += timed_run(model, data)
            num_runs += 1
        return elapsed_time_seconds, num_runs

    def run_for_iterations(model: transformers.BertModel, data: torch.Tensor, iterations: int):
        elapsed_time_seconds = 0.
        for _ in range(iterations):
            log.debug(f'Running iteration {_}, elapsed time {elapsed_time_seconds}')
            elapsed_time_seconds += timed_run(model, data)

        return elapsed_time_seconds, iterations


    if benchmark_time_seconds is not None and benchmark_iterations is not None:
        raise ValueError(f'Run time and number of iterations cannot be specified at the same time.')

    if benchmark_time_seconds is None and benchmark_iterations is None:
        raise ValueError(f'Either run time or number of iterations must be specified.')

    batch_size = data.shape[0]

    if benchmark_time_seconds is not None:
        # Warmup runs
        warmup_time_seconds = 0.1 * benchmark_time_seconds \
            if not warmup_time_seconds else warmup_time_seconds

        log.info(f'Running warmup cycles for {warmup_time_seconds} seconds.')
        run_for_seconds(model, data, warmup_time_seconds)

        # Benchmark runs
        log.info(f'Running benchmark cycles for {benchmark_time_seconds} seconds.')
        total_time_seconds, num_runs = run_for_seconds(
            model, data, benchmark_time_seconds)

    else:
        # Warmup runs
        warmup_iterations = ceil(0.1 * benchmark_iterations) \
            if not warmup_iterations else warmup_iterations

        log.info(f'Running warmup cycles for {warmup_iterations} iterations.')
        run_for_iterations(model, data, warmup_iterations)

        # Benchmark runs
        log.info(f'Running benchmark cycles for {benchmark_iterations} iterations.')
        total_time_seconds, num_runs = run_for_iterations(
            model, data, benchmark_iterations)

    average_time = total_time_seconds / num_runs
    average_throughput = batch_size / average_time

    return average_throughput, average_time


def vanilla_model(args):
    log.info(f'Preparing vanilla \'{args.model}\'')

    config = transformers.BertConfig.from_pretrained(
        args.model)
    model = transformers.BertModel.from_pretrained(
        args.model, config=config)
    model.eval()

    data = get_data(model, args)

    with torch.no_grad():
        model = torch.jit.trace(model, data, strict=False)
        model = torch.jit.freeze(model)
        model(data)
        model(data)

    model.config = config  # we restore the config after tracing so
    # that get_data() can read the vocab size and
    # max seq len from the model

    return model


def ipex_model(args):
    log.info(f'Preparing IPEX-optimized \'{args.model}\'')
    if args.quantization or args.bfloat16:
        raise NotImplementedError(
            'Quantization and BFloat16 not yet supported by the IPEX benchmark.')

    import intel_extension_for_pytorch as ipex
    config = transformers.BertConfig.from_pretrained(
        args.model)
    model = transformers.BertModel.from_pretrained(
        args.model, config=config)
    model.eval()

    data = get_data(model, args)

    model = ipex.optimize(model, dtype=torch.float32,
                          level="O1", auto_kernel_selection=True)

    with torch.no_grad():
        model = torch.jit.trace(model, data, strict=False)
        model = torch.jit.freeze(model)
        # Run inference twice to initialize the optimizations
        model(data)
        model(data)

    model.config = config  # we restore the config after tracing so
    # that get_data() can read the vocab size and
    # max seq len from the model

    return model


def bert_op_model(args):
    log.info(f'Preparing \'{args.model}\' using the BERT Op')
    import bert_op
    config = transformers.BertConfig.from_pretrained(
        args.model)
    config.use_quantization = args.quantization
    config.use_bfloat16 = args.bfloat16
    if config.use_quantization:
        config.quantization_factors = np.tile(np.tile([-10., 10.], 4),   # min/max values for one layer
                                              config.num_hidden_layers)  # repeat for number of layers
    model = transformers.BertModel.from_pretrained(
        args.model, config=config)
    model.eval()


    data = get_data(model, args)
    with torch.no_grad():
        model = torch.jit.trace(model, data, strict=False)
        model = torch.jit.freeze(model)
        # Run inference twice to initialize the optimizations
        model(data)
        model(data)
    
    model.config = config  # we restore the config after tracing so
    # that get_data() can read the vocab size and
    # max seq len from the model

    return model


def ipex_bert_op_model(args):
    raise NotImplementedError('BERT Op + IPEX is not yet implemented.')
    # TODO: (krzychut): RuntimeError: Cannot serialize custom bound C++ class __torch__.torch.classes.bert_op.BertOp. Please define serialization methods via def_pickle() for this class.
    model = bert_op_model(args)
    import intel_extension_for_pytorch as ipex

    model = ipex.optimize(model)

    return model


def get_data(model, args):
    vocab_size = model.config.vocab_size
    data = torch.randint(vocab_size, size=[args.batch_size, args.seq_len])
    return data


model_func = {
    (False, False): vanilla_model,
    (True, False): ipex_model,
    (False, True): bert_op_model,
    (True, True): ipex_bert_op_model
}


def main(args):
    model = model_func[(args.ipex, args.bert_op)](args)
    data = get_data(model, args)

    results = pd.DataFrame(
        columns=['Model',
                 'IPEX',
                 'BERT Op',
                 'Quantization',
                 'BFloat16',
                 'Batch Size',
                 'Seq Len',
                 'Throughput [samples/s]',
                 'Latency [ms]'])

    throughput, latency = run_benchmark(
        model, data, args.run_time, args.warmup_time, args.iterations, args.warmup_iterations)

    row = pd.Series({
        'Model': args.model,
        'IPEX': str(args.ipex),
        'BERT Op': str(args.bert_op),
        'Quantization': str(args.quantization),
        'BFloat16': str(args.bfloat16),
        'Batch Size': args.batch_size,
        'Seq Len': args.seq_len,
        'Throughput [samples/s]': f'{throughput:.3f}',
        'Latency [ms]': f'{latency * 1000 :.3f} ms'
    })
    results = pd.concat([results, row.to_frame().T], ignore_index=True)

    print(results.to_markdown())

    if args.csv is not None:

        # Try to merge with an existing data file
        try:
            data_in_file = pd.read_csv(args.csv, sep=args.sep)
            results = pd.concat([data_in_file, results])
        
        # This is fine, there may not have been a file to merge with
        except FileNotFoundError as e:
            pass
        
        results.to_csv(args.csv, sep=args.sep, index=False) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='BERT benchmarking script for comparing BertOp vs vanilla/IPEX performance.')

    parser.add_argument('-m', '--model', type=str, help='BERT model to load')

    parser.add_argument('--ipex', action='store_true',
                        default=False, help='Use IPEX')
    parser.add_argument('--bert-op', action='store_true',
                        default=False, help='Use the monolithic BERT op')

    parser.add_argument('-q', '--quantization', action='store_true',
                        default=False, help='Use quantization')
    parser.add_argument('-b', '--bfloat16', action='store_true',
                        default=False, help='Use BFloat16 operations')

    parser.add_argument('-B', '--batch-size', type=int,
                        default=1, help='Batch size to benchmark')
    parser.add_argument('-s', '--seq-len', type=int,
                        default=128, help='Sequence length of the data')

    parser.add_argument('-r', '--run-time', type=float,
                        help='Time in seconds to run the benchmark for. Cannot be used together with --iterations')
    parser.add_argument('-w', '--warmup-time', type=float,
                        help='Time in seconds to run the warmup for before the benchmark; Defaults to 10%% of --run-time')

    parser.add_argument('--iterations', type=int,
                        help='Number of iterations to run the benchmark for. Cannot be used toegher with --run-time')
    parser.add_argument('--warmup-iterations', type=int,
                        help='Number of iterations to run the warmup for before the benchmark. Defaults to 10%% of --iterations')

    parser.add_argument('--log', dest='log_level', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL', 'FATAL'],
                        default='WARN', help='Logging verbosity level')

    parser.add_argument('--output-csv', dest='csv', type=str, help='If provided, results will be saved to this .csv file.')
    parser.add_argument('--sep', default='\t', help='Separator used when writing to the .csv file.')

    args, _ = parser.parse_known_args()

    log.setLevel(args.log_level)

    main(args)
