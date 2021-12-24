// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_COMMON__
#define __DNNL_COMMON__

#include "dnnl.hpp"

#include <string>
#include <unordered_map>


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

template <typename T_input, typename T_wei, typename T_output, typename T_bias = float>
std::stringstream KeyConstructionInternal(std::string func_name, T_bias* bias = nullptr) {
    char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
    char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
    char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';


    std::stringstream weights_addr;
    weights_addr << func_name << '-' << type_input << type_weights << type_output;
    if(bias) {
        char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
        weights_addr << type_bias;
    }

    return weights_addr;
}

template <typename T_input, typename T_wei, typename T_output, typename T_bias = float>
std::string KeyConstruction(T_input* input, T_wei* weight, T_output* output,int m, int n,int k, std::string func_name,T_bias* bias = nullptr) {
    const void *address = static_cast<const void*>(weight);

    auto weights_addr = KeyConstructionInternal<T_input, T_wei, T_output, T_bias>(func_name, bias);
    weights_addr << '-' << m << '-' << n << '-' << k << '-' << address;
    return weights_addr.str();
}

template <typename T_input, typename T_wei, typename T_output, typename T_bias = float>
std::string KeyConstruction(T_input* input, T_wei* weight, T_output* output,int m, int n, std::string func_name,T_bias* bias = nullptr) {
    const void *address = static_cast<const void*>(weight);

    auto weights_addr = KeyConstructionInternal<T_input, T_wei, T_output, T_bias>(func_name, bias);
    weights_addr << '-' << m << '-' << n << '-' << address;
    return weights_addr.str();
}

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
