# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import tensorflow as tf
import sys
import argparse
import run_classifier

if (not tf.__version__.startswith('2')):
    print("This script currently doesn't support tf1")
    exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launches run_classifier.py from AI zoo after loading custom operator')
    parser.add_argument('path_to_bertop', help='path to custom bert operator')
    args, _ = parser.parse_known_args()
    tf.load_op_library(args.path_to_bertop)
    run_classifier.main('')
