FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

# Install the basics
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        lsb-release \
        build-essential \
        ca-certificates \
        cmake \
        ninja-build \
        git \
        gnupg2 \
        python3 \
        python3-dev \
        python3-distutils \
        python3-pip \
        python-is-python3 \
        wget


# The protobuf compiler is needed for the custom protos used by our project
RUN apt-get install -y \
        protobuf-compiler \
        libprotobuf-dev \
        --no-install-recommends


# For the oneAPI kit, we get what we can from binaries...
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O - | apt-key add -

RUN echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt-get update && \
    apt-get install -y \
        intel-oneapi-tbb \
        intel-oneapi-mkl \
        intel-oneapi-mkl-devel \
        intel-oneapi-tbb-devel \
        intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic


# ...but we must build oneDNN from source, since we need v2.7
ARG DNNL_BRANCH=rls-v2.7
WORKDIR /tmp/oneapi_src
RUN git clone --depth 1 --branch $DNNL_BRANCH https://github.com/oneapi-src/oneDNN.git oneDNN

SHELL ["bash", "-c"]

ARG DNNL_CPU_RUNTIME=OMP
WORKDIR /tmp/oneapi_src/oneDNN/build
RUN source /opt/intel/oneapi/setvars.sh intel64 && \
    cmake -DCMAKE_CXX_COMPILER=icpx \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DCMAKE_C_COMPILER=icx \
        -DDNNL_CPU_RUNTIME=$DNNL_CPU_RUNTIME \
        -DDNNL_LIBRARY_TYPE=SHARED \
        -DDNNL_GPU_RUNTIME=NONE \
        -G "Ninja" \
        .. && \
    ninja dnnl install

ENV DEMO_DIR=samples/tensorflow_no_model_modifier_performance

# Install the python requirements needed to run the benchmark
COPY $DEMO_DIR/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Cache the BERT-large model
RUN python -c 'import transformers; transformers.TFBertModel.from_pretrained("bert-large-uncased")'


# Install numactl
RUN apt-get update && \
    apt-get install -y \
        numactl \
        --no-install-recommends

# Compile the project
COPY . /libraries.ai.performance.models.bert
WORKDIR /libraries.ai.performance.models.bert/build
RUN . /opt/intel/oneapi/setvars.sh && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DBACKENDS=TF \
        ..
RUN . /opt/intel/oneapi/setvars.sh && \
    cmake --build . -j

ENV BERT_OP_LIB=/libraries.ai.performance.models.bert/build/src/tf_op/libBertOp.so
ENV PYTHONPATH=/libraries.ai.performance.models.bert/python

WORKDIR /
COPY $DEMO_DIR/entrypoint.sh entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
