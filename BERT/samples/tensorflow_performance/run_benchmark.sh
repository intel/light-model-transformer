#!/bin/bash

set -e

# Source the oneAPI components we installed from binaries
. /opt/intel/oneapi/setvars.sh --force

# Set up the environment to use the BERT op library
export BERT_OP_LIB=$(realpath $(find ../.. -name libBertOp.so))


ln -fs original_saved_model.pb bert-large/saved_model.pb
echo '-----------------------------------------------'
echo '| Running bert-large-uncased from HuggingFace |'
echo '-----------------------------------------------'
python benchmark.py -b $1 -w $2 -i $3 bert-large
echo '------------------------------------------------'
echo '| FINISHED bert-large-uncased from HuggingFace |'
echo '------------------------------------------------'

# Now symlink the optimized model graph to compare
# the performance to baseline.
ln -fs modified_saved_model.pb bert-large/saved_model.pb
echo '-------------------------------------------------------------------------'
echo '| Running bert-large-uncased from HuggingFace with a monolithic BERT op |'
echo '-------------------------------------------------------------------------'
python benchmark.py -b $1 -w $2 -i $3 bert-large
echo '--------------------------------------------------------------------------'
echo '| FINISHED bert-large-uncased from HuggingFace with a monolithic BERT op |'
echo '--------------------------------------------------------------------------'
