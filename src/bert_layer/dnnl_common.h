// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_COMMON__
#define __DNNL_COMMON__

#include "dnnl.hpp"

#include <sstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <sstream>

using bfloat16 = std::uint16_t;

template <class T> struct DnnlDataType;

template <> struct DnnlDataType<float> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::f32;
};

template <> struct DnnlDataType<bfloat16> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::bf16;
};

template <> struct DnnlDataType<int8_t> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::s8;
};

template <> struct DnnlDataType<uint8_t> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::u8;
};

typedef std::unordered_map<std::string, dnnl::memory> map_mem_t;
typedef std::unordered_map<std::string, dnnl::inner_product_forward::primitive_desc> map_ip_primd_t;
typedef std::unordered_map<std::string, dnnl::matmul::primitive_desc> map_mm_primd_t;
typedef std::unordered_map<std::string, dnnl::batch_normalization_forward::primitive_desc> map_bn_primd_t;
typedef std::unordered_map<std::string, dnnl::layer_normalization_forward::primitive_desc> map_ln_primd_t;
typedef std::unordered_map<std::string, dnnl::primitive> map_prim_t;

class DnnlCommon {
    public:
    DnnlCommon() : eng(dnnl::engine::kind::cpu, 0), eng_stream(eng) {}

    dnnl::stream& getEngineStream(){
        return eng_stream;
    };

    dnnl::engine& getEngine(){
        return eng;
    }

    map_mem_t& get_g_memory(){
      return g_memory;
    }

    map_mm_primd_t& get_g_mm_prim_desc(){
      return g_mm_prim_desc;
    }

    map_ln_primd_t& get_g_ln_prim_desc(){
      return g_ln_prim_desc;
    }
  
    map_prim_t& get_g_prim(){
      return g_prim;
    }
    
    private:
    dnnl::engine eng;
    dnnl::stream eng_stream;
    map_mem_t g_memory;
    map_mm_primd_t g_mm_prim_desc;
    map_ln_primd_t g_ln_prim_desc;
    map_prim_t g_prim;
};

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

#endif
