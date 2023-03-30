#!/bin/bash


. /opt/intel/oneapi/setvars.sh

set -e

cd /libraries.ai.performance.models.bert/tests/pytorch


before=()
after=()
i=1
for var; do
    i=$(($i + 1))
    if [[ $var == '--' ]]; then
        break
    fi
    before+=($var)
done

for var in ${@:$i}; do
    after+=($var)
done

${before[*]} ./benchmark.py ${after[*]}
