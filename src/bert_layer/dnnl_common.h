// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_COMMON__
#define __DNNL_COMMON__

#include "dnnl.hpp"

#include <string>
#include <sstream> // TODO(rbogdano): Delete this include after cleanup commit merge.

using bfloat16 = std::uint16_t;


typedef std::unordered_map<std::string, dnnl::memory*> map_mem_t;
typedef std::unordered_map<std::string, dnnl::inner_product_forward::primitive_desc*> map_ip_primd_t;
typedef std::unordered_map<std::string, dnnl::matmul::primitive_desc*> map_mm_primd_t;
typedef std::unordered_map<std::string, dnnl::batch_normalization_forward::primitive_desc*> map_bn_primd_t;
typedef std::unordered_map<std::string, dnnl::layer_normalization_forward::primitive_desc*> map_ln_primd_t;
typedef std::unordered_map<std::string, dnnl::primitive*> map_prim_t;

dnnl::engine eng(dnnl::engine::kind::cpu, 0);

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

    for (map_bn_primd_t::iterator iter = g_bn_prim_desc.begin(); iter != g_bn_prim_desc.end(); ++iter) {
      delete iter->second;
    }

    for (map_ln_primd_t::iterator iter = g_ln_prim_desc.begin(); iter != g_ln_prim_desc.end(); ++iter) {
      delete iter->second;
    }

    for (map_prim_t::iterator iter = g_prim.begin(); iter != g_prim.end(); ++iter) {
      delete iter->second;
    }
}

#endif
