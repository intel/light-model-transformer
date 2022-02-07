// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_SOFTMAX__
#define __DNNL_SOFTMAX__

#include "dnnl_common.h"
#include "dnnl_attr.hpp"
#include "dnnl_data.hpp"
#include "dnnl_ops.hpp"

template <typename T_input>
bool Softmax(DnnlCommon& dnnl_context, T_input* input, int m, int n) {
    using namespace dnnl_wrappers;

    auto& eng = dnnl_context.getEngine();
    auto& stm = dnnl_context.getEngineStream();

    auto prim_key = KeyConstruction(input,input,input,m,n,"Softmax");

    dnnl::memory::dims data_tz = {m, n};

    auto data_memory = AttachMemory(eng, data_tz, input);
    auto input_data = DataSource(data_memory);

    const int axis = 1;
    auto softmax = CachedSoftmax<float>(prim_key, dnnl_context, m, n, axis);
    softmax.Compute(stm, input_data, data_memory);
    return true;
}

#endif
