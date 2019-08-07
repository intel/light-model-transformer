# What is it?

A more efficient implementation for the standard BERT.

## Steps to verify

- Install Intel(R) MKL
- Install Intel(R) Compiler (Optional, but it may achieve better performance)
- `cd tf_ops && sh compile.sh` to compile the source code, please note: if Intel(R) Compiler does not exist, please modify the script (you may need to run `source /opt/intel/mkl/bin/mklvars.sh intel64` to set MKL path before the compiling)
- `python modify_model.py bert_token128.pb bert_token128_fused.pb`, to fuse all BERT ops to a single 'Bert' op, it will generate bert_toker128_fused.pb (For your own model, you may need modify the script to do model transformation)
- Run the new solution by `source env.sh && numactl --cpunodebind=0 --membind=0 python pb_inference.py bert_token128_fused.pb loss/Softmax`
