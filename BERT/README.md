# What is it?

A more efficient implementation for the base BERT (originated from https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip).

## Prepare

- Install Intel(R) MKL
- Install tensorflow
- Install Intel(R) Compiler (Optional, but it may achieve better performance)
- Download the reference model: https://github.com/intel/light-model-transformer/releases/download/bert0.1/bert_token128.pb

## Steps to verify
- Compile the source code:
```
source /opt/intel/mkl/bin/mklvars.sh intel64
cd tf_ops && sh compile.sh
```
(please modify the script to switch between ICC and GCC)
- Fuse all ops in BERT layers to a single 'Bert' op. Following command will transform bert_token128.pb to bert_toker128_fused.pb (For your own model, you may need modify the script to do model transformation)
```
python modify_model.py bert_token128.pb bert_token128_fused.pb
```
- Run the new solution like following command:
```
source env.sh && numactl --cpunodebind=0 --membind=0 python pb_inference.py bert_token128_fused.pb loss/Softmax
```
