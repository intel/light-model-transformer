# BERT-large demo of the monolithic BERT operator

## Introduction

This is a sample used to demonstrate performance of the BERT operator optimization.
The sample will download a fresh `bert-large-uncased` model from HuggingFace and
convert it to SavedModel format.

A short benchmark script will then be run. First, the script will generate an optimized
saved_model.pb which uses the monolithic BERT operator. This may take a few minutes.
Next, the model will be tested on dummy data, first using the original model graph,
and then using the optimized graph. The benchmark uses a fixed sequence length of 128,
though the batch size and the number of warmup/benchmark iterations can be configured.

The next section provides instructions on how to run the sample in a Docker container.

The third section provides instructions on how to run the sample on a clean Ubuntu 20.04
environment (e.g. an AWS instance), though this was not tested. The steps essentially
involve running the scripts the same way they are executed in the [Dockerfile](Dockerfile).

## Docker demo:

1. Build the docker image by running the utility script in the `samples/tensorflow_performance` directory:
    ```bash
    ./build_image -t <image-tag> <any other docker build args here>
    ```
    **NOTE:** It may be necessary to change pass proxy setup to `docker build`, the [Dockerfile](Dockerfile) accepts
    build arguments for this purpose, for example:
    ```bash
    ./build_image -t <image-tag> --build-arg http_proxy=<http-proxy> --build-arg https_proxy=<https_proxy>
    ```

2. Run the benchmark inside a container:
    ```bash
    docker run <image-tag> BATCH_SIZE WARMUP_ITERATIONS BENCHMARK_ITERATIONS
    ```
    For example:
    ```bash
    docker run <image-tag> 16 10 50
    ```

## Bare-metal / AWS instance demo:

From the `tensorflow_performance` directory:

1. Set up proxy variables if necessary, for example:
    ``` bash
    export http_proxy=<http_proxy>
    export https_proxy=<https_proxy>
    export no_proxy=<no_proxy>
    echo "Acquire::http::proxy \"${http_proxy}\";" >> /etc/apt/apt.conf
    ```

2. Install dependencies:
    ```bash
    ./install_dependencies.sh
    ```

3. Compile the project:
    ```bash
    ./compile.sh
    ```

4. Prepare the model:
    ```bash
    ./prepare_model.sh
    ```

5. Run the benchmark:
    ```bash
    ./run_benchmark BATCH_SIZE WARMUP_ITERATIONS BENCHMARK_ITERATIONS
    ```
    For example:
    ```bash
    ./run_benchmark 16 10 50
    ```
