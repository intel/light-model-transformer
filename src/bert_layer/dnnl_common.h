// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_COMMON__
#define __DNNL_COMMON__

#include "dnnl.hpp"
#include "dnnl_debug.h"

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>

/*
#if defined(ONEDNN_2_2)
#include "common/bfloat16.hpp"
extern "C" {
dnnl_status_t DNNL_API dnnl_gemm_bf16bf16f32(char transa, char transb,
         dnnl_dim_t M, dnnl_dim_t N, dnnl_dim_t K, float alpha,
         const dnnl::impl::bfloat16_t *A, dnnl_dim_t lda,
         const dnnl::impl::bfloat16_t *B, dnnl_dim_t ldb, float beta,
         float *C, dnnl_dim_t ldc);
}
#elif defined(ONEDNN_1_3)
#include "bfloat16.hpp"
#endif

typedef dnnl::impl::bfloat16_t bfloat16;
*/

using bfloat16 = std::uint16_t;

using namespace dnnl;

typedef std::unordered_map<std::string, memory*> map_mem_t;
typedef std::unordered_map<std::string, inner_product_forward::primitive_desc*> map_ip_primd_t;
typedef std::unordered_map<std::string, matmul::primitive_desc*> map_mm_primd_t;
typedef std::unordered_map<std::string, batch_normalization_forward::primitive_desc*> map_bn_primd_t;
typedef std::unordered_map<std::string, layer_normalization_forward::primitive_desc*> map_ln_primd_t;
typedef std::unordered_map<std::string, primitive*> map_prim_t;

engine eng(engine::kind::cpu, 0);

map_mem_t g_memory;
map_ip_primd_t g_ip_prim_desc;
map_mm_primd_t g_mm_prim_desc;
map_bn_primd_t g_bn_prim_desc;
map_ln_primd_t g_ln_prim_desc;
map_prim_t g_prim;

void del_dnnl(void)
{
    for (map_mem_t::iterator iter = g_memory.begin(); iter != g_memory.end(); ++iter) {
      delete iter->second;
    }

    for (map_ip_primd_t::iterator iter = g_ip_prim_desc.begin(); iter != g_ip_prim_desc.end(); ++iter) {
      delete iter->second;
    }

    for (map_mm_primd_t::iterator iter = g_mm_prim_desc.begin(); iter != g_mm_prim_desc.end(); ++iter) {
      delete iter->second;
    }

    for (map_prim_t::iterator iter = g_prim.begin(); iter != g_prim.end(); ++iter) {
      delete iter->second;
    }
}

#endif
