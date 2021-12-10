// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_MATMUL_QUANT__
#define __DNNL_MATMUL_QUANT__

#include "dnnl_common.h"

#include <sstream>
#include <string>
#include <iostream>
#include <type_traits>


#define src_format dnnl::memory::format_tag::ab
#define bias_format dnnl::memory::format_tag::ab
#define dst_format dnnl::memory::format_tag::ab
#define weights_format dnnl::memory::format_tag::any

#define user_src_format dnnl::memory::format_tag::ab
#define user_bias_format dnnl::memory::format_tag::ab
#define user_weights_format dnnl::memory::format_tag::ab
#define user_weights_format_trans dnnl::memory::format_tag::ba

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_erf_quant(dnnl::engine eng, dnnl::stream stm, T_input* input, T_wei* weight, T_bias* bias, T_output* output, 
        int m, int n, int k, bool wTrans, float src_scale, float weight_scale) {
    char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
    char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
    char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
    char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

    const void *address = static_cast<const void*>(weight);

    std::stringstream weights_addr;
    weights_addr << "MatMul_with_erf_quant-" << type_input << type_weights << type_bias << type_output \
                 << '-' << m << '-' << n << '-' << k << '-' << address;
    std::string prim_key = weights_addr.str();

    dnnl::memory::dims src_tz = { m, k };
    dnnl::memory::dims weights_tz = { k, n };
    dnnl::memory::dims bias_tz = { 1, n };
    dnnl::memory::dims dst_tz = { m, n };
   
    dnnl::memory::data_type src_dt = dnnl::memory::data_type::s8;
    dnnl::memory::data_type weights_dt = dnnl::memory::data_type::s8;
    dnnl::memory::data_type bias_dt = dnnl::memory::data_type::f32;
    dnnl::memory::data_type dst_dt = dnnl::memory::data_type::f32;

    auto it_prim_created = g_prim.find(prim_key);
    if (it_prim_created == g_prim.end())
    {
        auto src_md     = dnnl::memory::desc({ src_tz }, src_dt, src_format);
        auto weights_md = dnnl::memory::desc({ weights_tz }, weights_dt, weights_format);
        auto bias_md    = dnnl::memory::desc({ bias_tz }, bias_dt, bias_format);
        auto dst_md     = dnnl::memory::desc({ dst_tz }, dst_dt, dst_format);  

        auto desc = dnnl::matmul::desc(src_md, weights_md, bias_md, dst_md);

        dnnl::primitive_attr attr;
        dnnl::post_ops po;

        const int dst_mask = 0;
        attr.set_output_scales(dst_mask, {1/(src_scale * weight_scale)});

        po.append_eltwise(
            1.0f, //scale
            dnnl::algorithm::eltwise_gelu_erf,
            0.f, // neg slope
            0.f  /*unused for relu */ );

        attr.set_post_ops(po);

        auto *prim_desc = new dnnl::matmul::primitive_desc(desc, attr, eng);
        auto *prim = new dnnl::matmul(*prim_desc);

        g_prim.insert(std::pair<std::string, dnnl::primitive *>(prim_key, prim));
        g_mm_prim_desc.insert(std::pair<std::string, dnnl::matmul::primitive_desc *>(prim_key, prim_desc));
    }

    auto user_src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, user_src_format);
    auto user_weights_md = dnnl::memory::desc(weights_tz, dnnl::memory::data_type::f32, user_weights_format); // ab or ba
    if (wTrans) {
        user_weights_md = dnnl::memory::desc(weights_tz, dnnl::memory::data_type::f32, user_weights_format_trans);
    }
    auto user_bias_md = dnnl::memory::desc(bias_tz, dnnl::memory::data_type::f32, user_bias_format);

    auto user_src_memory = dnnl::memory(user_src_md, eng, input);
    auto user_weights_memory = dnnl::memory(user_weights_md, eng, weight);
    auto user_bias_memory = dnnl::memory(user_bias_md, eng, bias);

    auto it_prim_desc_created = g_mm_prim_desc.find(prim_key);
    if (it_prim_desc_created == g_mm_prim_desc.end()) {
        std::cout << "MatMul_with_erf error: can find g_mm_prim_desc = " << prim_key << std::endl;
        return false;
    }
    dnnl::matmul::primitive_desc prim_desc = *it_prim_desc_created->second;

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;

    dnnl::primitive_attr src_attr;
    dnnl::primitive_attr weights_attr;
    dnnl::primitive_attr bias_attr;
    src_attr.set_output_scales(src_mask, {src_scale});
    weights_attr.set_output_scales(weight_mask, {weight_scale});
    bias_attr.set_output_scales(bias_mask, {src_scale * weight_scale});

    auto src_memory = user_src_memory;
    auto weights_memory = &user_weights_memory;
    auto bias_memory = &user_bias_memory;
    auto dst_memory = dnnl::memory(prim_desc.dst_desc(), eng, output);

    if (prim_desc.src_desc() != user_src_memory.get_desc()) {
        src_memory = dnnl::memory(prim_desc.src_desc(), eng);
        auto reorder_src = dnnl::reorder(user_src_memory, src_memory, src_attr);
        reorder_src.execute(stm, {
            { DNNL_ARG_FROM, user_src_memory },
            { DNNL_ARG_TO, src_memory } });
    }

    if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        std::string prim_weights_key = prim_key+"-weights";
        auto it_memory_created = g_memory.find(prim_weights_key);

        if (it_memory_created == g_memory.end()) {
            weights_memory = new dnnl::memory(prim_desc.weights_desc(), eng);
            auto reorder_weights = dnnl::reorder(user_weights_memory, *weights_memory, weights_attr);
            reorder_weights.execute(stm, {
                { DNNL_ARG_FROM, user_weights_memory },
                { DNNL_ARG_TO, *weights_memory } });
            g_memory.insert(std::pair<std::string, dnnl::memory *>(prim_weights_key, weights_memory));
        }
        else {
            weights_memory = it_memory_created->second;
        }
    }

    {
        std::string prim_bias_key = prim_key+"-bias";
        auto it_memory_created = g_memory.find(prim_bias_key);

        if (it_memory_created == g_memory.end()) {
            bias_memory = new dnnl::memory(prim_desc.bias_desc(), eng);
            auto reorder_bias = dnnl::reorder(user_bias_memory, *bias_memory, bias_attr);
            reorder_bias.execute(stm, {
                { DNNL_ARG_FROM, user_bias_memory },
                { DNNL_ARG_TO, *bias_memory } });
            g_memory.insert(std::pair<std::string, dnnl::memory *>(prim_bias_key, bias_memory));
        }
        else {
            bias_memory = it_memory_created->second;
        }
    }

    it_prim_created = g_prim.find(prim_key);
    if (it_prim_created != g_prim.end()) {
        it_prim_created->second->execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_WEIGHTS, *weights_memory },
            { DNNL_ARG_BIAS, *bias_memory },
            { DNNL_ARG_DST, dst_memory } });
    }
    else {
        std::cout << "MatMul_with_erf: execute error, prim_key = " << prim_key << std::endl;
        return false;
    }

    stm.wait();
    return true;
}


template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_sum_quant(dnnl::engine eng, dnnl::stream stm, T_input* input, T_wei* weight, T_bias* bias, T_output* output, 
        int m, int n, int k, bool wTrans, float src_scale, float weight_scale) {
    char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
    char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
    char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
    char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

    const void *address = static_cast<const void*>(weight);

    std::stringstream weights_addr;
    weights_addr << "MatMul_with_sum-" << type_input << type_weights << type_bias << type_output \
                 << '-' << m << '-' << n << '-' << k << '-' << address;
    std::string prim_key = weights_addr.str();

    dnnl::memory::dims src_tz = { m, k };
    dnnl::memory::dims weights_tz = {k, n };
    dnnl::memory::dims bias_tz = { 1, n };
    dnnl::memory::dims dst_tz = {m, n };
   
    dnnl::memory::data_type src_dt = dnnl::memory::data_type::s8;
    dnnl::memory::data_type weights_dt = dnnl::memory::data_type::s8;
    dnnl::memory::data_type bias_dt = dnnl::memory::data_type::f32;
    dnnl::memory::data_type dst_dt = dnnl::memory::data_type::f32;

    auto it_prim_created = g_prim.find(prim_key);

    if (it_prim_created == g_prim.end()) {
        auto src_md     = dnnl::memory::desc({ src_tz }, src_dt, src_format);
        auto weights_md = dnnl::memory::desc({ weights_tz }, weights_dt, weights_format);
        auto bias_md    = dnnl::memory::desc({ bias_tz }, bias_dt, bias_format);
        auto dst_md     = dnnl::memory::desc({ dst_tz }, dst_dt, dst_format); 

        auto desc = dnnl::matmul::desc(src_md, weights_md, bias_md, dst_md);

        float beta = 1.0f;
       
        dnnl::primitive_attr attr;
        dnnl::post_ops po;

        const int dst_mask = 0;
        attr.set_output_scales(dst_mask, {1/(src_scale * weight_scale)});

        po.append_sum(beta);
        attr.set_post_ops(po);

        auto *prim_desc = new dnnl::matmul::primitive_desc(desc, attr, eng);
        auto *prim = new dnnl::matmul(*prim_desc);

        g_prim.insert(std::pair<std::string, dnnl::primitive *>(prim_key, prim));
        g_mm_prim_desc.insert(std::pair<std::string, dnnl::matmul::primitive_desc *>(prim_key, prim_desc));
    }

    auto user_src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, user_src_format);
    auto user_weights_md = dnnl::memory::desc(weights_tz, dnnl::memory::data_type::f32, user_weights_format); // ab or ba
    if (wTrans) {
        user_weights_md = dnnl::memory::desc(weights_tz, dnnl::memory::data_type::f32, user_weights_format_trans);
    }
    auto user_bias_md = dnnl::memory::desc(bias_tz, dnnl::memory::data_type::f32, user_bias_format);

    auto user_src_memory = dnnl::memory(user_src_md, eng, input);
    auto user_weights_memory = dnnl::memory(user_weights_md, eng, weight);
    auto user_bias_memory = dnnl::memory(user_bias_md, eng, bias);

    auto it_prim_desc_created = g_mm_prim_desc.find(prim_key);
    if (it_prim_desc_created == g_mm_prim_desc.end()) {
        std::cout << "MatMul_with_sum error: can find g_mm_prim_desc = " << prim_key << std::endl;
        return false;
    }
    dnnl::matmul::primitive_desc prim_desc = *it_prim_desc_created->second;

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;

    dnnl::primitive_attr src_attr;
    dnnl::primitive_attr weights_attr;
    dnnl::primitive_attr bias_attr;
    src_attr.set_output_scales(src_mask, {src_scale});
    weights_attr.set_output_scales(weight_mask, {weight_scale});
    bias_attr.set_output_scales(bias_mask, {src_scale * weight_scale});

    auto src_memory = user_src_memory;
    auto weights_memory = &user_weights_memory;
    auto bias_memory = &user_bias_memory;
    auto dst_memory = dnnl::memory(prim_desc.dst_desc(), eng, output);

    if (prim_desc.src_desc() != user_src_memory.get_desc()) {
        src_memory = dnnl::memory(prim_desc.src_desc(), eng);
        auto reorder_src = dnnl::reorder(user_src_memory, src_memory, src_attr);
        reorder_src.execute(stm, {
            { DNNL_ARG_FROM, user_src_memory },
            { DNNL_ARG_TO, src_memory } });
    }

    if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        std::string prim_weights_key = prim_key+"-weights";
        auto it_memory_created = g_memory.find(prim_weights_key);

        if (it_memory_created == g_memory.end()) {
            weights_memory = new dnnl::memory(prim_desc.weights_desc(), eng);
            auto reorder_weights = dnnl::reorder(user_weights_memory, *weights_memory, weights_attr);
            reorder_weights.execute(stm, {
                { DNNL_ARG_FROM, user_weights_memory },
                { DNNL_ARG_TO, *weights_memory } });
            g_memory.insert(std::pair<std::string, dnnl::memory *>(prim_weights_key, weights_memory));
        }
        else {
            weights_memory = it_memory_created->second;
        }
    }

    {
        std::string prim_bias_key = prim_key+"-bias";
        auto it_memory_created = g_memory.find(prim_bias_key);

        if (it_memory_created == g_memory.end()) {
            bias_memory = new dnnl::memory(prim_desc.bias_desc(), eng);
            auto reorder_bias = dnnl::reorder(user_bias_memory, *bias_memory, bias_attr);
            reorder_bias.execute(stm, {
                { DNNL_ARG_FROM, user_bias_memory },
                { DNNL_ARG_TO, *bias_memory } });
            g_memory.insert(std::pair<std::string, dnnl::memory *>(prim_bias_key, bias_memory));
        }
        else {
            bias_memory = it_memory_created->second;
        }
    }

    it_prim_created = g_prim.find(prim_key);
    if (it_prim_created != g_prim.end()) {
        it_prim_created->second->execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_WEIGHTS, *weights_memory },
            { DNNL_ARG_BIAS, *bias_memory },
            { DNNL_ARG_DST, dst_memory } });
    }
    else {
        std::cout << "MatMul_with_sum: execute error, prim_key = " << prim_key << std::endl;
        return false;
    }

    stm.wait();
    return true;
}

template <typename T_input, typename T_wei, typename T_bias, typename T_output>
bool MatMul_with_bias_quant(dnnl::engine eng, dnnl::stream stm, T_input* input, T_wei* weight, T_bias* bias, T_output* output, 
        int m, int n, int k, bool wTrans, float src_scale, float weight_scale) {
    char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
    char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
    char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
    char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

    const void *address = static_cast<const void*>(weight);

    std::stringstream weights_addr;
    weights_addr << "MatMul_with_bias-" << type_input << type_weights << type_bias << type_output \
                 << '-' << m << '-' << n << '-' << k << '-' << address;
    std::string prim_key = weights_addr.str();

    dnnl::memory::dims src_tz = { m, k };
    dnnl::memory::dims weights_tz = {k, n };
    dnnl::memory::dims bias_tz = { 1, n };
    dnnl::memory::dims dst_tz = {m, n };
   
    dnnl::memory::data_type src_dt = dnnl::memory::data_type::s8;
    dnnl::memory::data_type weights_dt = dnnl::memory::data_type::s8;
    dnnl::memory::data_type bias_dt = dnnl::memory::data_type::f32;
    dnnl::memory::data_type dst_dt = dnnl::memory::data_type::f32;

    const int dst_mask = 0;
    dnnl::primitive_attr dst_attr;
    dst_attr.set_output_scales(dst_mask, {1/(src_scale * weight_scale)});

    auto it_prim_created = g_prim.find(prim_key);
    if (it_prim_created == g_prim.end())
    {
        auto src_md     = dnnl::memory::desc({ src_tz }, src_dt, src_format);
        auto weights_md = dnnl::memory::desc({ weights_tz }, weights_dt, weights_format);
        auto bias_md    = dnnl::memory::desc({ bias_tz }, bias_dt, bias_format);
        auto dst_md     = dnnl::memory::desc({ dst_tz }, dst_dt, dst_format);  

        auto desc = dnnl::matmul::desc(src_md, weights_md, bias_md, dst_md);

        auto *prim_desc = new dnnl::matmul::primitive_desc(desc, dst_attr, eng);
        auto *prim = new dnnl::matmul(*prim_desc);

        g_prim.insert(std::pair<std::string, dnnl::primitive *>(prim_key, prim));
        g_mm_prim_desc.insert(std::pair<std::string, dnnl::matmul::primitive_desc *>(prim_key, prim_desc));
    }

    auto user_src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, user_src_format);
    auto user_weights_md = dnnl::memory::desc(weights_tz, dnnl::memory::data_type::f32, user_weights_format); // ab or ba
    if (wTrans) {
        user_weights_md = dnnl::memory::desc(weights_tz, dnnl::memory::data_type::f32, user_weights_format_trans);
    }
    auto user_bias_md = dnnl::memory::desc(bias_tz, dnnl::memory::data_type::f32, user_bias_format);

    auto user_src_memory = dnnl::memory(user_src_md, eng, input);
    auto user_weights_memory = dnnl::memory(user_weights_md, eng, weight);
    auto user_bias_memory = dnnl::memory(user_bias_md, eng, bias);

    auto it_prim_desc_created = g_mm_prim_desc.find(prim_key);
    if (it_prim_desc_created == g_mm_prim_desc.end()) {
        std::cout << "MatMul_with_bias error: can find g_mm_prim_desc = " << prim_key << std::endl;
        return false;
    }
    dnnl::matmul::primitive_desc prim_desc = *it_prim_desc_created->second;

    const int src_mask = 0;
    const int weight_mask = 0;
    const int bias_mask = 0;

    dnnl::primitive_attr src_attr;
    dnnl::primitive_attr weights_attr;
    dnnl::primitive_attr bias_attr;
    src_attr.set_output_scales(src_mask, {src_scale});
    weights_attr.set_output_scales(weight_mask, {weight_scale});
    bias_attr.set_output_scales(bias_mask, {src_scale * weight_scale});

    auto src_memory = user_src_memory;
    auto weights_memory = &user_weights_memory;
    auto bias_memory = &user_bias_memory;
    auto dst_memory = dnnl::memory(prim_desc.dst_desc(), eng, output);

    if (prim_desc.src_desc() != user_src_memory.get_desc()) {
        src_memory = dnnl::memory(prim_desc.src_desc(), eng);
        auto reorder_src = dnnl::reorder(user_src_memory, src_memory, src_attr);
        reorder_src.execute(stm, {
            { DNNL_ARG_FROM, user_src_memory },
            { DNNL_ARG_TO, src_memory } });
    }

    if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        std::string prim_weights_key = prim_key+"-weights";
        auto it_memory_created = g_memory.find(prim_weights_key);

        if (it_memory_created == g_memory.end()) {
            weights_memory = new dnnl::memory(prim_desc.weights_desc(), eng);
            auto reorder_weights = dnnl::reorder(user_weights_memory, *weights_memory, weights_attr);
            reorder_weights.execute(stm, {
                { DNNL_ARG_FROM, user_weights_memory },
                { DNNL_ARG_TO, *weights_memory } });
            g_memory.insert(std::pair<std::string, dnnl::memory *>(prim_weights_key, weights_memory));
        }
        else {
            weights_memory = it_memory_created->second;
        }
    }

    {
        std::string prim_bias_key = prim_key+"-bias";
        auto it_memory_created = g_memory.find(prim_bias_key);

        if (it_memory_created == g_memory.end()) {
            bias_memory = new dnnl::memory(prim_desc.bias_desc(), eng);
            auto reorder_bias = dnnl::reorder(user_bias_memory, *bias_memory, bias_attr);
            reorder_bias.execute(stm, {
                { DNNL_ARG_FROM, user_bias_memory },
                { DNNL_ARG_TO, *bias_memory } });
            g_memory.insert(std::pair<std::string, dnnl::memory *>(prim_bias_key, bias_memory));
        }
        else {
            bias_memory = it_memory_created->second;
        }
    }

    it_prim_created = g_prim.find(prim_key);
    if (it_prim_created != g_prim.end()) {
        it_prim_created->second->execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_WEIGHTS, *weights_memory },
            { DNNL_ARG_BIAS, *bias_memory },
            { DNNL_ARG_DST, dst_memory } });
    }
    else {
        std::cout << "MatMul_with_bias: execute error, prim_key = " << prim_key << std::endl;
        return false;
    }

    stm.wait();
    return true;
}

#endif
