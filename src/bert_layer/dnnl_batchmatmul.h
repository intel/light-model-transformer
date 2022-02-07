// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_BATCH_MATMUL__
#define __DNNL_BATCH_MATMUL__

#include "dnnl_common.h"
#include "dnnl_attr.hpp"
#include "dnnl_data.hpp"
#include "dnnl_ops.hpp"

#include <string>
#include <sstream>
#include <type_traits>

dnnl::memory::desc StridedMD(int batch, int x, int y, int batch_stride, int ld, dnnl::memory::data_type dt, bool trans = false) {
    const dnnl::memory::dims tz{batch, x, y};
    const auto stride = trans ? dnnl::memory::dims{batch_stride, 1, ld} : dnnl::memory::dims{batch_stride, ld, 1};
    return dnnl::memory::desc{tz, dt, stride};
}

template <typename T_input, typename T_wei, typename T_output, typename T_bias>
bool BatchMatMul(const std::string& prim_key, DnnlCommon& dnnl_context,
                 T_input* input, T_wei* weight, T_output* output, T_bias* bias,
                 dnnl::memory::desc src_md, dnnl::memory::desc weights_md, dnnl::memory::desc dst_md, dnnl::memory::desc bias_md,
        const dnnl_wrappers::BuildAttrs& attr = {}) {
    using namespace dnnl_wrappers;

    auto eng = dnnl_context.getEngine();
    auto stm = dnnl_context.getEngineStream();
    auto& g_mem = dnnl_context.get_g_memory();

    auto mat_mul = CachedMatMul(prim_key, dnnl_context, src_md, weights_md, bias_md, dst_md, attr);

    // hack MD data types for data sources
    src_md.data.data_type = dnnl::memory::convert_to_c(DnnlDataType<T_input>::value);
    weights_md.data.data_type = dnnl::memory::convert_to_c(DnnlDataType<T_wei>::value);
    bias_md.data.data_type = dnnl::memory::convert_to_c(DnnlDataType<T_bias>::value);

    auto input_data   = DataSource(dnnl::memory{src_md, eng, input});
    auto weights_data = GCachedDataSource(prim_key + "-weights", g_mem, dnnl::memory{weights_md, eng, weight});
    
    auto user_bias_mem = bias ? dnnl::memory{bias_md, eng, bias} : dnnl::memory{};
    auto bias_data    = GCachedDataSource(prim_key + "-bias", g_mem, user_bias_mem);

    auto output_memory = dnnl::memory(dst_md, eng, output);

    mat_mul.Compute(stm, input_data, weights_data, bias_data, output_memory);

    return true;
}

template <typename T_input, typename T_wei, typename T_output, typename T_bias>
bool BatchMatMul_with_stride_bias(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_output* output, T_bias* bias,
        int m, int n, int k, int lda, int ldb, int ldc, 
        int batch_stride_a, int batch_stride_b, int batch_stride_c, int batch_stride_bias,
        int batch_src, int batch_weights, int batch_dst, int batch_bias, float scale, bool wTrans) {

    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"BatchMatMul_with_stride_bias",bias);

#if 0
    dnnl::memory::data_type src_dt = dnnl::memory::data_type::bf16;
    dnnl::memory::data_type weights_dt = dnnl::memory::data_type::bf16;
    dnnl::memory::data_type bias_dt = dnnl::memory::data_type::f32;
#else
    dnnl::memory::data_type src_dt = dnnl::memory::data_type::f32;
    dnnl::memory::data_type weights_dt = dnnl::memory::data_type::f32;
    dnnl::memory::data_type bias_dt = dnnl::memory::data_type::f32;
#endif

    dnnl::memory::data_type dst_dt = DnnlDataType<T_output>::value;

    auto src_md     = StridedMD(batch_src, m, k, batch_stride_a, lda, src_dt);
    auto weights_md = StridedMD(batch_weights, k, n, batch_stride_b, ldb, weights_dt, wTrans);
    auto bias_md    = StridedMD(batch_bias, 1, n, batch_stride_bias, 0, bias_dt);
    auto dst_md     = StridedMD(batch_dst, m, n, batch_stride_c, ldc, dst_dt);

    return BatchMatMul(prim_key, dnnl_context, input, weight, output, bias,
                       src_md, weights_md, dst_md, bias_md, dnnl_wrappers::BuildAttrs().Scale(scale));
}

template <typename T_input, typename T_wei, typename T_output>
bool BatchMatMul_with_stride(DnnlCommon& dnnl_context, T_input* input, T_wei* weight, T_output* output, 
        int m, int n, int k, int lda, int ldb, int ldc, 
        int batch_stride_a, int batch_stride_b, int batch_stride_c, bool wTrans, int batch) {
    
    auto prim_key = KeyConstruction(input,weight,output,m,n,k,"BatchMatMul_with_stride");

#if 0
    dnnl::memory::data_type src_dt = dnnl::memory::data_type::bf16;
    dnnl::memory::data_type weights_dt = dnnl::memory::data_type::bf16;
#else
    dnnl::memory::data_type src_dt = dnnl::memory::data_type::f32;
    dnnl::memory::data_type weights_dt = dnnl::memory::data_type::f32;
#endif

    dnnl::memory::data_type dst_dt = DnnlDataType<T_output>::value;

    auto src_md     = StridedMD(batch, m, k, batch_stride_a, lda, src_dt);
    auto weights_md = StridedMD(batch, k, n, batch_stride_b, ldb, weights_dt, wTrans);
    auto dst_md     = StridedMD(batch, m, n, batch_stride_c, ldc, dst_dt);

    return BatchMatMul(prim_key, dnnl_context, input, weight, output, (float*)nullptr,
                       src_md, weights_md, dst_md, {});
}

#endif
