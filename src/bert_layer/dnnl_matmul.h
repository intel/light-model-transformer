// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_MATMUL__
#define __DNNL_MATMUL__

#include "dnnl_common.h"
#include "dnnl_attr.hpp"
#include "dnnl_data.hpp"
#include "dnnl_ops.hpp"

#include <sstream>
#include <string>
#include <iostream>
#include <type_traits>

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_noquant(const std::string& prim_key, DnnlCommon& dnnl_context,
                    T_input* input, T_wei* weight, T_bias* bias, T_output* output,
                    int m, int n, int k, bool wTrans, const dnnl_wrappers::BuildAttrs& attr = {}) {
    using namespace dnnl_wrappers;

    auto& eng = dnnl_context.getEngine();
    auto& stm = dnnl_context.getEngineStream();
    auto& g_mem = dnnl_context.get_g_memory();

    const MatMulDims dims{m, n, k};

    auto input_data = DataSource(AttachMemory(eng, dims.src_tz, input));
    auto weights_data = GCachedDataSource(prim_key + "-weights", g_mem, AttachMemory(eng, dims.weights_tz, weight, wTrans));
    auto bias_data = GCachedDataSource(prim_key + "-bias", g_mem, AttachMemory(eng, dims.bias_tz, bias));
    auto output_memory = AttachMemory(eng, dims.dst_tz, output);

#if 1
    auto mat_mul = CachedMatMul<bfloat16, bfloat16, T_bias, T_output>(prim_key, dnnl_context, m, n, k, attr);
#else
    auto mat_mul = CachedMatMul<T_input, T_wei, T_bias, T_output>(prim_key, dnnl_context, m, n, k, attr);
#endif

    mat_mul.Compute(stm, input_data, weights_data, bias_data, output_memory);

    return true;
}

// FIXME(rfsaliev) defines below never used but kept for simplified diff in review
#define src_format dnnl::memory::format_tag::ab
#define bias_format dnnl::memory::format_tag::ab
#define dst_format dnnl::memory::format_tag::ab
#define weights_format dnnl::memory::format_tag::any

#define user_src_format dnnl::memory::format_tag::ab
#define user_bias_format dnnl::memory::format_tag::ab
#define user_weights_format dnnl::memory::format_tag::ab
#define user_weights_format_trans dnnl::memory::format_tag::ba

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_erf_dst_bf16(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k, bool wTrans) {
    
    using namespace dnnl_wrappers;

    auto& eng = dnnl_context.getEngine();
    auto& stm = dnnl_context.getEngineStream();
    auto& g_mem = dnnl_context.get_g_memory();

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_erf_dst_bf16",bias);

    const MatMulDims dims{m, n, k};

    auto input_data = DataSource(AttachMemory(eng, dims.src_tz, input));
    auto weights_data = GCachedDataSource(prim_key + "-weights", g_mem, AttachMemory(eng, dims.weights_tz, weight, wTrans));
    auto bias_data = GCachedDataSource(prim_key + "-bias", g_mem, AttachMemory(eng, dims.bias_tz, bias));
    auto output_memory = AttachMemory(eng, dims.dst_tz, output);

    dnnl::primitive_attr attr = BuildAttrs().Eltwise(dnnl::algorithm::eltwise_gelu_erf);

    auto mat_mul = CachedMatMul<bfloat16, bfloat16, bfloat16, T_output>(prim_key, dnnl_context, m, n, k, attr);
    mat_mul.Compute(stm, input_data, weights_data, bias_data, output_memory);

    return true;
}

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_erf_src_bf16(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k, bool wTrans)
{
    // Expected T_input == bfloat16
    static_assert(DnnlDataType<T_input>::value == dnnl::memory::data_type::bf16);
    // The rest of logic similar to MatMul_with_erf<>() which supports any type T_input
    // FIXME(rfsaliev) MatMul_with_erf<>() is not defined yet - copy-paste for now
    using namespace dnnl_wrappers;

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_erf_src_bf16",bias);

    return MatMul_noquant(prim_key, dnnl_context,
                          input, weight, bias, output,
                          m, n, k, wTrans,
                          BuildAttrs().Eltwise(dnnl::algorithm::eltwise_gelu_erf));
}

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_erf(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k, bool wTrans) {
    using namespace dnnl_wrappers;

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_erf",bias);

    return MatMul_noquant(prim_key, dnnl_context,
                          input, weight, bias, output,
                          m, n, k, wTrans,
                          BuildAttrs().Eltwise(dnnl::algorithm::eltwise_gelu_erf));
}

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_sum(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k, bool wTrans) {
    using namespace dnnl_wrappers;

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_sum",bias);

    float beta = 1.0f;
    return MatMul_noquant(prim_key, dnnl_context,
                          input, weight, bias, output,
                          m, n, k, wTrans,
                          BuildAttrs().Sum(beta));
}

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_sum_src_bf16(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k, bool wTrans) {
    // Expected T_input == bfloat16
    static_assert(DnnlDataType<T_input>::value == dnnl::memory::data_type::bf16);
    // FIXME(rfsaliev) The rest of logic similar to MatMul_with_sum<>() which supports any type T_input except prim_key
    using namespace dnnl_wrappers;

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_sum_src_bf16",bias);

    float beta = 1.0f;
    return MatMul_noquant(prim_key, dnnl_context,
                          input, weight, bias, output,
                          m, n, k, wTrans,
                          BuildAttrs().Sum(beta));
}


template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_bias(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_bias* bias, T_output* output, int m, int n, int k, bool wTrans) {
    using namespace dnnl_wrappers;

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"MatMul_with_bias",bias);

    return MatMul_noquant(prim_key, dnnl_context,
                          input, weight, bias, output,
                          m, n, k, wTrans);
}

#endif
