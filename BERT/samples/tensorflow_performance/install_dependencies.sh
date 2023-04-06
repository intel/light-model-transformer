#!/bin/bash

set -e


export DEBIAN_FRONTEND=noninteractive

apt-get update

# Install the basics
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

apt-get clean all


# The protobuf compiler is needed for the custom protos used by our project
apt-get install -y \
        protobuf-compiler \
        libprotobuf-dev \
        --no-install-recommends


# For the oneAPI kit, we get what we can from binaries...
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O - | apt-key add -

echo "deb https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
apt-get update
apt-get install -y \
    intel-oneapi-tbb \
    intel-oneapi-tbb-devel \
    intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic


tmp_dir=$(mktemp -d)
trap "rm -rf $tmp_dir" EXIT

# ...but we must build oneDNN from source, since we need v2.7
DNNL_BRANCH=rls-v2.7
pushd $tmp_dir
git clone --depth 1 --branch $DNNL_BRANCH https://github.com/oneapi-src/oneDNN.git oneDNN

mkdir build && cd build
source /opt/intel/oneapi/setvars.sh --force intel64 && \
    cmake -DCMAKE_CXX_COMPILER=icpx \
          -DCMAKE_C_COMPILER=icx \
          -DDNNL_CPU_RUNTIME=TBB \
          -DDNNL_LIBRARY_TYPE=SHARED \
          -DDNNL_GPU_RUNTIME=NONE \
          -H$tmp_dir/oneDNN \
          -G "Ninja" && \
    ninja dnnl install

popd
rm -rf $tmp_dir

# Finally we install the python requirements needed to run the benchmark
pip install -r requirements.txt
