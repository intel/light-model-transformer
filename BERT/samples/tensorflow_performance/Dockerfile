FROM ubuntu:focal as base

ARG DEBIAN_FRONTEND=noninteractive

ENV PROJECT_DIR=bert_optimization
ENV SAMPLE_DIR=samples/tensorflow_performance

ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy=""
RUN test ${http_proxy} && echo "Acquire::http::proxy \"${http_proxy}\";" >> /etc/apt/apt.conf

SHELL ["/bin/bash", "-c"]

# Copy just the requirements first and isntall dependencies.
# This avoids rebuilding this step when unrelated file changes are made.
WORKDIR ${PROJECT_DIR}/${SAMPLE_DIR}
COPY ${SAMPLE_DIR}/install_dependencies.sh install_dependencies.sh
COPY ${SAMPLE_DIR}/requirements.txt requirements.txt
RUN ./install_dependencies.sh

# Now copy the rest of the project
WORKDIR /
COPY . bert_optimization
WORKDIR ${PROJECT_DIR}/${SAMPLE_DIR}
RUN ./compile.sh

RUN ./prepare_model.sh

RUN apt update && apt install -y numactl --no-install-recommends

ENTRYPOINT ["/bin/bash", "run_benchmark.sh"] CMD ['1', '20', '100']
