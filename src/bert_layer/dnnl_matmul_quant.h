// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_MATMUL_QUANT__
#define __DNNL_MATMUL_QUANT__

#include "dnnl_common.h"
#include "dnnl_attr.hpp"
#include "dnnl_data.hpp"
#include "dnnl_ops.hpp"

#include <sstream>
#include <string>
#include <iostream>
#include <type_traits>

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_quant(const std::string& prim_key, DnnlCommon& dnnl_context,
                  T_input* input, T_wei* weight, T_bias* bias, T_output* output,
                  int m, int n, int k, bool wTrans,
                  float src_scale, float weight_scale, dnnl_wrappers::BuildAttrs attr = {}) {
    using namespace dnnl_wrappers;

    auto& eng = dnnl_context.getEngine();
    auto& stm = dnnl_context.getEngineStream();
    auto& g_mem = dnnl_context.get_g_memory();

    const MatMulDims dims{m, n, k};

    auto output_scale = 1/(src_scale * weight_scale);
    auto bias_scale = src_scale * weight_scale;

    auto input_data = ScaledData(AttachMemory(eng, dims.src_tz, input), src_scale);
    auto weights_data = ScaledCachedData(prim_key + "-weights", g_mem, AttachMemory(eng, dims.weights_tz, weight, wTrans), weight_scale);
    auto bias_data = ScaledCachedData(prim_key + "-bias", g_mem, AttachMemory(eng, dims.bias_tz, bias), bias_scale);
    auto output_memory = AttachMemory(eng, dims.dst_tz, output);

    auto mat_mul = CachedMatMul<int8_t, int8_t, float, T_output>(prim_key, dnnl_context, m, n, k, attr.Scale(output_scale));
    mat_mul.Compute(stm, input_data, weights_data, bias_data, output_memory);

    return true;
}

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_erf_quant(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, 
        int m, int n, int k, bool wTrans, float src_scale, float weight_scale) {
    using namespace dnnl_wrappers;

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_erf_quant",bias);
    return MatMul_quant(prim_key, dnnl_context,
                        input, weight, bias, output,
                        m, n, k, wTrans,
                        src_scale, weight_scale,
                        BuildAttrs().Eltwise(dnnl::algorithm::eltwise_gelu_erf));
}

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_sum_quant(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, 
        int m, int n, int k, bool wTrans, float src_scale, float weight_scale) {
    using namespace dnnl_wrappers;

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_sum_quant",bias);
    return MatMul_quant(prim_key, dnnl_context,
                        input, weight, bias, output,
                        m, n, k, wTrans,
                        src_scale, weight_scale,
                        BuildAttrs().Sum());
}

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_bias_quant(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, 
        int m, int n, int k, bool wTrans, float src_scale, float weight_scale) {
    using namespace dnnl_wrappers;

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_bias_quant",bias);
    return MatMul_quant(prim_key, dnnl_context,
                        input, weight, bias, output,
                        m, n, k, wTrans,
                        src_scale, weight_scale);
}

#endif
