#!/bin/bash

set -e

# Source the oneAPI components we installed from binaries
. /opt/intel/oneapi/setvars.sh --force

# Generate a saved model from the HuggingFace repository
python hf_to_saved_model.py bert-large-uncased bert-large

# Set up the environment to use the model optimization tool
# and the BERT op library
export BERT_OP_LIB=$(realpath $(find ../.. -name libBertOp.so))
export PYTHONPATH=$PYTHONPATH:$(realpath ../../python)

# Optimize the model graph (an optimized copy will be saved to
# bert-large/modified_saved_model.pb)
./replace_full_bert.sh bert-large bert-large

# Preserve the original model graph. We can then use symlinks to easily switch
# between the original and optimized graphs
cd bert-large
mv saved_model.pb original_saved_model.pb

# Symlink the optimized graph into place for a moment. We need to configure
# the BERT op nodes for FP32 mode with max sequence length of 128
ln -fs modified_saved_model.pb saved_model.pb
python -m model_modifier.configure_bert_op -B -Q -s 128 .

# Now the setup is done, so symlink the original model
# back into place to run the baseline benchmark.
ln -fs original_saved_model.pb saved_model.pb
