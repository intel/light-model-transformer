# BERT-op TensorFlow demo

## Introduction

This demo can be used to measure BertOp optimization performance on TensorFlow BERT models. It uses
a similar approach to the PyTorch optimization, i.e. it does not require use of the model modifier tool.

## Usage

1. Build Docker image

To build the demo image, use the utility script:
```bash
./build_demo.sh
```

Optionally, add any build arguments you want, which will be forwarded to `docker build`, for example:
```bash
./build_demo.sh -t <image-tag> --build-arg DNNL_CPU_RUNTIME=TBB
```

2. Run the benchmark

To run the benchmark, execute the container with the desired parameters. Two parameter sets are accepted:
* before "`--`" - these will be prepended to the `benchmark.py` execution - put things like `numactl` here
* after "`--`" - these will passed to `benchmark.py` - use them to configure the model (`--help` to list available params)

```bash
# --privileged may be needed e.g. to use `--membind` of numactl
docker run --privileged <image-tag> <EXECUTION ARGS> -- <BENCHMARK ARGS>
```

For example:
```
docker run --privileged <image-tag> numactl --cpunodebind=1 --membind=1 -- -m bert-large-uncased --bert-op --bf16 --run-time 60
```
