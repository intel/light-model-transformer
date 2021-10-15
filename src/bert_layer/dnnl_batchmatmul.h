// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_BATCH_MATMUL__
#define __DNNL_BATCH_MATMUL__

#include "dnnl_common.h"

#define batch_src_format memory::format_tag::abc
#define batch_bias_format memory::format_tag::abc
#define batch_dst_format memory::format_tag::abc
#define batch_weights_format memory::format_tag::any

#define batch_user_src_format memory::format_tag::abc
#define batch_user_bias_format memory::format_tag::abc
#define batch_user_weights_format memory::format_tag::abc
#define batch_user_weights_format_trans memory::format_tag::acb


template <typename T_input, typename T_wei, typename T_output, typename T_bias>
bool BatchMatMul_with_stride_bias(engine eng, stream stm, T_input* input, T_wei* weight, T_output* output, T_bias* bias,
        int m, int n, int k, int lda, int ldb, int ldc, 
        int batch_stride_a, int batch_stride_b, int batch_stride_c, int batch_stride_bias,
        int batch_src, int batch_weights, int batch_dst, int batch_bias, float scale, bool wTrans)
{
    char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
    char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
    char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
    char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

    const void *address = static_cast<const void*>(weight);

    std::stringstream weights_addr;
    weights_addr << "BatchMatMul_with_stride_bias-" << type_input << type_weights << type_output << type_bias \
                 << '-' << m << '-' << n << '-' << k << '-' << address;
    std::string prim_key = weights_addr.str();

    memory::dims src_tz = { batch_src, m, k };
    memory::dims weights_tz = {batch_weights, k, n };
    memory::dims dst_tz = {batch_dst, m, n };
    memory::dims bias_tz = {batch_bias, 1, n };

    memory::dims a_stride = memory::dims {batch_stride_a, lda, 1};
    memory::dims b_stride = wTrans ? memory::dims {batch_stride_b, 1, ldb} : memory::dims {batch_stride_b, ldb, 1};
    memory::dims c_stride = memory::dims {batch_stride_c, ldc, 1};
    memory::dims bias_stride = memory::dims {batch_stride_bias, 0, 1};

#if 0
    memory::data_type src_dt = memory::data_type::bf16;
    memory::data_type weights_dt = memory::data_type::bf16;
    memory::data_type dst_dt = memory::data_type::f32;
    memory::data_type bias_dt = memory::data_type::f32;
#else
    memory::data_type src_dt = memory::data_type::f32;
    memory::data_type weights_dt = memory::data_type::f32;
    memory::data_type dst_dt = memory::data_type::f32;
    memory::data_type bias_dt = memory::data_type::f32;
#endif

    auto it_prim_created = g_prim.find(prim_key);
    if (it_prim_created == g_prim.end())
    {
        auto src_md     = memory::desc({ src_tz }, src_dt, a_stride);
        auto weights_md = memory::desc({ weights_tz }, weights_dt, b_stride);

        auto bias_md    = memory::desc({ bias_tz }, bias_dt, bias_stride);
        auto dst_md     = memory::desc({ dst_tz }, dst_dt, c_stride);  

        auto desc = matmul::desc(src_md, weights_md, bias_md, dst_md);
        //auto desc = matmul::desc(src_md, weights_md, dst_md);

#if 1
        //float scale = 0.125f;
        primitive_attr attr;
        attr.set_output_scales(/* mask */ 0, {scale});

        auto *prim_desc = new matmul::primitive_desc(desc, attr, eng);
#else
        auto *prim_desc = new matmul::primitive_desc(desc, eng);
#endif
        auto *prim = new matmul(*prim_desc);

        g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
        g_mm_prim_desc.insert(std::pair<std::string, matmul::primitive_desc *>(prim_key, prim_desc));
        //std::cout << "MatMul_with_bias: save prim_key = " << prim_key << ", prim number = " << g_prim.size() << std::endl;
    }

    auto user_src_md = memory::desc(src_tz, memory::data_type::f32, a_stride);
    auto user_weights_md = memory::desc(weights_tz, memory::data_type::f32, b_stride);
    auto user_bias_md = memory::desc(bias_tz, memory::data_type::f32, bias_stride);

    auto user_src_memory = memory(user_src_md, eng, input);
    auto user_weights_memory = memory(user_weights_md, eng, weight);
    auto user_bias_memory = memory(user_bias_md, eng, bias);

    auto it_prim_desc_created = g_mm_prim_desc.find(prim_key);
    if (it_prim_desc_created == g_mm_prim_desc.end()) {
        std::cout << "can find g_mm_prim_desc = " << prim_key << std::endl;
        return false;
    }
    matmul::primitive_desc prim_desc = *it_prim_desc_created->second;

    auto src_memory = user_src_memory;
    auto weights_memory = &user_weights_memory;
    auto bias_memory = &user_bias_memory;
    auto dst_memory = memory(prim_desc.dst_desc(), eng, output);

    if (prim_desc.src_desc() != user_src_memory.get_desc()) {
        #if 0
        static int index = 0;
        index++;
        if (index < 2)
            std::cout << "reorder user_src_memory !!!" << std::endl;
        #endif

        src_memory = memory(prim_desc.src_desc(), eng);
        auto reorder_src = reorder(user_src_memory, src_memory);
        reorder_src.execute(stm, {
            { DNNL_ARG_FROM, user_src_memory },
            { DNNL_ARG_TO, src_memory } });
    }

    if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        std::string prim_weights_key = prim_key+"-weights";
        auto it_memory_created = g_memory.find(prim_weights_key);
        if (it_memory_created == g_memory.end()) {

            //std::cout << "MatMul: reorder user_weights_memory !!!" << std::endl;
            weights_memory = new memory(prim_desc.weights_desc(), eng);
            auto reorder_weights = reorder(user_weights_memory, *weights_memory);
            reorder_weights.execute(stm, {
                { DNNL_ARG_FROM, user_weights_memory },
                { DNNL_ARG_TO, *weights_memory } });
            g_memory.insert(std::pair<std::string, memory *>(prim_weights_key, weights_memory));
        }
        else {
            weights_memory = it_memory_created->second;
        }
    }


    if (prim_desc.bias_desc() != user_bias_memory.get_desc()) {
        std::string prim_bias_key = prim_key+"-bias";
        auto it_memory_created = g_memory.find(prim_bias_key);
        if (it_memory_created == g_memory.end()) {
            //std::cout << "MatMul: reorder user_bias_memory !!!" << std::endl;
            bias_memory = new memory(prim_desc.bias_desc(), eng);
            auto reorder_bias = reorder(user_bias_memory, *bias_memory);
            reorder_bias.execute(stm, {
                { DNNL_ARG_FROM, user_bias_memory },
                { DNNL_ARG_TO, *bias_memory } });
            g_memory.insert(std::pair<std::string, memory *>(prim_bias_key, bias_memory));
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
        std::cout << "execute error, prim_key = " << prim_key << std::endl;
        return false;
    }
    stm.wait();
    return true;
}

template <typename T_input, typename T_wei, typename T_output>
bool BatchMatMul_with_stride(engine eng, stream stm, T_input* input, T_wei* weight, T_output* output, 
        int m, int n, int k, int lda, int ldb, int ldc, 
        int batch_stride_a, int batch_stride_b, int batch_stride_c, bool wTrans, int batch)
{
    char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
    char type_weights = (std::is_floating_point<T_wei>::value) ? 'f' : 'b';
    //char type_bias = (std::is_floating_point<T_bias>::value) ? 'f' : 'b';
    char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

    const void *address = static_cast<const void*>(weight);

    std::stringstream weights_addr;
    weights_addr << "BatchMatMul_with_stride-" << type_input << type_weights << type_output \
                 << '-' << m << '-' << n << '-' << k << '-' << address;
    std::string prim_key = weights_addr.str();

    memory::dims src_tz = { batch, m, k };
    memory::dims weights_tz = {batch, k, n };
    memory::dims dst_tz = {batch, m, n };

    memory::dims a_stride = memory::dims {batch_stride_a, lda, 1};
    memory::dims b_stride = wTrans ? memory::dims {batch_stride_b, 1, ldb} : memory::dims {batch_stride_b, ldb, 1};
    memory::dims c_stride = memory::dims {batch_stride_c, ldc, 1};

#if 0
    memory::data_type src_dt = memory::data_type::bf16;
    memory::data_type weights_dt = memory::data_type::bf16;
    memory::data_type dst_dt = memory::data_type::f32;
#else
    memory::data_type src_dt = memory::data_type::f32;
    memory::data_type weights_dt = memory::data_type::f32;
    memory::data_type dst_dt = memory::data_type::f32;
#endif

    auto it_prim_created = g_prim.find(prim_key);
    if (it_prim_created == g_prim.end())
    {
        auto src_md     = memory::desc({ src_tz }, src_dt, a_stride);
        auto weights_md = memory::desc({ weights_tz }, weights_dt, b_stride);

        //auto bias_md    = memory::desc({ bias_tz }, bias_dt, batch_bias_format);
        auto dst_md     = memory::desc({ dst_tz }, dst_dt, c_stride);  

        //auto desc = matmul::desc(src_md, weights_md, bias_md, dst_md);
        auto desc = matmul::desc(src_md, weights_md, dst_md);

        auto *prim_desc = new matmul::primitive_desc(desc, eng);
        auto *prim = new matmul(*prim_desc);

        g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
        g_mm_prim_desc.insert(std::pair<std::string, matmul::primitive_desc *>(prim_key, prim_desc));
        //std::cout << "MatMul_with_bias: save prim_key = " << prim_key << ", prim number = " << g_prim.size() << std::endl;
    }

    auto user_src_md = memory::desc(src_tz, memory::data_type::f32, a_stride);
    auto user_weights_md = memory::desc(weights_tz, memory::data_type::f32, b_stride);

    auto user_src_memory = memory(user_src_md, eng, input);
    auto user_weights_memory = memory(user_weights_md, eng, weight);
    //auto user_bias_memory = memory(user_bias_md, eng, bias);

    auto it_prim_desc_created = g_mm_prim_desc.find(prim_key);
    if (it_prim_desc_created == g_mm_prim_desc.end()) {
        std::cout << "can find g_mm_prim_desc = " << prim_key << std::endl;
        return false;
    }
    matmul::primitive_desc prim_desc = *it_prim_desc_created->second;

    auto src_memory = user_src_memory;
    auto weights_memory = &user_weights_memory;
    //auto bias_memory = &user_bias_memory;
    auto dst_memory = memory(prim_desc.dst_desc(), eng, output);

    if (prim_desc.src_desc() != user_src_memory.get_desc()) {
        #if 0
        static int index = 0;
        index++;
        if (index < 2)
            std::cout << "reorder user_src_memory !!!" << std::endl;
        #endif

        src_memory = memory(prim_desc.src_desc(), eng);
        auto reorder_src = reorder(user_src_memory, src_memory);
        reorder_src.execute(stm, {
            { DNNL_ARG_FROM, user_src_memory },
            { DNNL_ARG_TO, src_memory } });
    }

    if (prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        std::string prim_weights_key = prim_key+"-weights";
        auto it_memory_created = g_memory.find(prim_weights_key);
        if (it_memory_created == g_memory.end()) {

            //std::cout << "MatMul: reorder user_weights_memory !!!" << std::endl;
            weights_memory = new memory(prim_desc.weights_desc(), eng);
            auto reorder_weights = reorder(user_weights_memory, *weights_memory);
            reorder_weights.execute(stm, {
                { DNNL_ARG_FROM, user_weights_memory },
                { DNNL_ARG_TO, *weights_memory } });
            g_memory.insert(std::pair<std::string, memory *>(prim_weights_key, weights_memory));
        }
        else {
            weights_memory = it_memory_created->second;
        }
    }

#if 0
    if (prim_desc.bias_desc() != user_bias_memory.get_desc()) {
        std::string prim_bias_key = prim_key+"-bias";
        auto it_memory_created = g_memory.find(prim_bias_key);
        if (it_memory_created == g_memory.end()) {
            //std::cout << "MatMul: reorder user_bias_memory !!!" << std::endl;
            bias_memory = new memory(prim_desc.bias_desc(), eng);
            auto reorder_bias = reorder(user_bias_memory, *bias_memory);
            reorder_bias.execute(stm, {
                { DNNL_ARG_FROM, user_bias_memory },
                { DNNL_ARG_TO, *bias_memory } });
            g_memory.insert(std::pair<std::string, memory *>(prim_bias_key, bias_memory));
        }
        else {
            bias_memory = it_memory_created->second;
        }
    }
#endif

    it_prim_created = g_prim.find(prim_key);
    if (it_prim_created != g_prim.end()) {
        it_prim_created->second->execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_WEIGHTS, *weights_memory },
            //{ DNNL_ARG_BIAS, *bias_memory },
            { DNNL_ARG_DST, dst_memory } });
    }
    else {
        std::cout << "execute error, prim_key = " << prim_key << std::endl;
        return false;
    }
    stm.wait();
    return true;
}

#endif
