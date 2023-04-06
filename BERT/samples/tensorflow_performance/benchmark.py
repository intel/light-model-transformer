import tensorflow as tf
import time

import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-i', '--iterations', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('-w', '--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('model', type=str, help='Saved model directory')

    args = parser.parse_args()

    tf.load_op_library(os.environ.get('BERT_OP_LIB'))
    model = tf.saved_model.load(args.model)
    
    batch_size = args.batch_size
    seq_len = 128
    input_word_ids = tf.random.uniform(shape=(batch_size, seq_len), minval=0, maxval=30522, dtype=tf.int32)
    input_type_ids = tf.zeros(shape=(batch_size, seq_len), dtype=tf.int32)
    input_mask = tf.ones(shape=(batch_size, seq_len), dtype=tf.int32)
    input = {"input_ids":input_word_ids, "token_type_ids": input_type_ids, "attention_mask":input_mask}
    
    print(f'Benchmark setup:')
    print(f'\tBatch size: {args.batch_size}')
    print(f'\tWarmup iterations: {args.warmup}')
    print(f'\tBenchmark iterations: {args.iterations}')
    
    print(f'Starting {args.warmup} warmup iterations.')
    for _ in range(args.warmup):
        res = model.signatures["serving_default"](**input)
    
    print(f'Starting {args.iterations} benchmark iterations.')
    start = time.time()
    for _ in range(args.iterations):
        res = model.signatures["serving_default"](**input)
    end = time.time()
    elapsed = end - start

    print('Benchmark complete!')
    print(f'Average latency: {elapsed / args.iterations * 1000.}ms')
    print(f'Average throughput: {args.iterations / elapsed * batch_size} samples/s')   
