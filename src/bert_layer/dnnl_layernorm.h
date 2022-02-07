// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_LAYERNORM__
#define __DNNL_LAYERNORM__

#include "dnnl_common.h"
#include "dnnl_data.hpp"
#include "dnnl_ops.hpp"

#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <memory>


template <typename T_input, typename T_gamma, typename T_beta>
bool LayerNorm_with_gamma_beta(DnnlCommon& dnnl_context, T_input* input, T_gamma* gamma, T_beta* beta, int m, int n) {
    using namespace dnnl_wrappers;

    auto& eng = dnnl_context.getEngine();
    auto& stm = dnnl_context.getEngineStream();

    auto prim_key = KeyConstruction(input,gamma,beta,m,n,"LayerNorm_with_gamma_beta");

    dnnl::memory::dims data_tz = {m, n};
    dnnl::memory::dims gamma_tz = {1, n};
    dnnl::memory::dims beta_tz = {1, n};

    auto data_memory = AttachMemory(eng, data_tz, input);
    auto input_data = DataSource(data_memory);
    auto gamma_data = DataSource(AttachMemory(eng, gamma_tz, gamma));
    auto beta_data  = DataSource(AttachMemory(eng, beta_tz, beta));

    const float epsilon = 9.999999960041972e-13;
    const dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift;

    auto layer_norm = CachedLayerNorm<float>(prim_key, dnnl_context, m, n, epsilon, flags);
    layer_norm.Compute(stm, input_data, gamma_data, beta_data, data_memory);
    return true;
}

#endif
