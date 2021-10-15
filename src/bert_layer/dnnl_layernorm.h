// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_LAYERNORM__
#define __DNNL_LAYERNORM__

#include "dnnl_common.h"

template <typename T_input, typename T_gamma, typename T_beta>
bool LayerNorm_with_gamma_beta(engine eng, stream stm, T_input* input, T_gamma* gamma, T_beta* beta, int m, int n)
{
    char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
    char type_gamma = (std::is_floating_point<T_gamma>::value) ? 'f' : 'b';
    char type_beta = (std::is_floating_point<T_beta>::value) ? 'f' : 'b';
    //char type_output = (std::is_floating_point<T_output>::value) ? 'f' : 'b';

    const void *address = static_cast<const void*>(gamma);

    std::stringstream gamma_addr;
    gamma_addr << "LayerNorm_with_gamma_beta-" << type_input << type_gamma << type_beta \
                 << '-' << m << '-' << n << '-' << address;
    std::string prim_key = gamma_addr.str();

    memory::dims src_tz = { m, n };
    memory::dims gamma_tz = {1, n};
    memory::dims beta_tz = {1, n};

    memory::data_type src_dt = memory::data_type::f32;
    memory::data_type gamma_dt = memory::data_type::f32;
    memory::data_type beta_dt = memory::data_type::f32;

    auto it_prim_created = g_prim.find(prim_key);
    if (it_prim_created == g_prim.end())
    {
        auto src_md     = memory::desc({ src_tz }, src_dt, memory::format_tag::nc);
        //auto gamma_md = memory::desc({ gamma_tz }, gamma_dt, memory::format_tag::nc);
        //auto beta_md = memory::desc({ beta_tz }, beta_dt, memory::format_tag::nc);

        auto bnorm_d = layer_normalization_forward::desc(
                prop_kind::forward_inference, src_md, 9.999999960041972e-13,
                normalization_flags::use_scale | normalization_flags::use_shift );

        // Create primitive descriptor.
        auto *prim_desc = new layer_normalization_forward::primitive_desc(bnorm_d, eng);
        auto *prim = new layer_normalization_forward(*prim_desc);

        g_prim.insert(std::pair<std::string, primitive *>(prim_key, prim));
        g_ln_prim_desc.insert(std::pair<std::string, layer_normalization_forward::primitive_desc *>(prim_key, prim_desc));
        //std::cout << "batchnorm: save prim_key = " << prim_key << ", prim number = " << g_prim.size() << std::endl;
    }

    auto user_src_md = memory::desc(src_tz, memory::data_type::f32, memory::format_tag::nc);
    auto user_gamma_md = memory::desc(gamma_tz, memory::data_type::f32, memory::format_tag::nc); // ab or ba
    auto user_beta_md = memory::desc(beta_tz, memory::data_type::f32, memory::format_tag::nc); 

    auto user_src_memory = memory(user_src_md, eng, input);
    auto user_gamma_memory = memory(user_gamma_md, eng, gamma);
    auto user_beta_memory = memory(user_beta_md, eng, beta);

    auto it_prim_desc_created = g_ln_prim_desc.find(prim_key);
    if (it_prim_desc_created == g_ln_prim_desc.end()) {
        std::cout << "batchnorm error: can find g_ln_prim_desc = " << prim_key << std::endl;
        return false;
    }
    layer_normalization_forward::primitive_desc prim_desc = *it_prim_desc_created->second;

    auto src_memory = user_src_memory;
    auto gamma_memory = &user_gamma_memory;
    auto beta_memory = &user_beta_memory;

    //auto mean_mem = memory(prim_desc.mean_desc(), eng);
    //auto variance_mem = memory(prim_desc.variance_desc(), eng);

    if (prim_desc.src_desc() != user_src_memory.get_desc()) {
        #if 0
        static int index = 0;
        index++;
        if (index < 2)
            std::cout << "batchnorm: reorder user_src_memory !!!" << std::endl;
        #endif

        src_memory = memory(prim_desc.src_desc(), eng);
        auto reorder_src = reorder(user_src_memory, src_memory);
        reorder_src.execute(stm, {
            { DNNL_ARG_FROM, user_src_memory },
            { DNNL_ARG_TO, src_memory } });
    }
#if 0
    if (prim_desc.scale_desc() != user_gamma_memory.get_desc()) {
        std::string prim_gamma_key = prim_key+"-gamma";
        auto it_memory_created = g_memory.find(prim_gamma_key);
        if (it_memory_created == g_memory.end()) {

            //std::cout << "batchnorm: reorder user_gamma_memory !!!" << std::endl;
            gamma_memory = new memory(prim_desc.scale_desc(), eng);
            auto reorder_gamma = reorder(user_gamma_memory, *gamma_memory);
            reorder_gamma.execute(stm, {
                { DNNL_ARG_FROM, user_gamma_memory },
                { DNNL_ARG_TO, *gamma_memory } });
            g_memory.insert(std::pair<std::string, memory *>(prim_gamma_key, gamma_memory));
        }
        else {
            gamma_memory = it_memory_created->second;
        }
    }

    if (prim_desc.shift_desc() != user_beta_memory.get_desc()) {
        std::string prim_beta_key = prim_key+"-beta";
        auto it_memory_created = g_memory.find(prim_beta_key);
        if (it_memory_created == g_memory.end()) {
            //std::cout << "batchnorm: reorder user_beta_memory !!!" << std::endl;
            beta_memory = new memory(prim_desc.shift_desc(), eng);
            auto reorder_beta = reorder(user_beta_memory, *beta_memory);
            reorder_beta.execute(stm, {
                { DNNL_ARG_FROM, user_beta_memory },
                { DNNL_ARG_TO, *beta_memory } });
            g_memory.insert(std::pair<std::string, memory *>(prim_beta_key, beta_memory));
        }
        else {
            beta_memory = it_memory_created->second;
        }
    }
#endif
    it_prim_created = g_prim.find(prim_key);
    if (it_prim_created != g_prim.end()) {
        it_prim_created->second->execute(stm, {
            //{ DNNL_ARG_MEAN, mean_mem },
            //{ DNNL_ARG_VARIANCE, variance_mem },
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_SCALE, *gamma_memory },
            { DNNL_ARG_SHIFT, *beta_memory },
            { DNNL_ARG_DST, src_memory } });
    }
    else {
        std::cout << "batchnorm: execute error, prim_key = " << prim_key << std::endl;
        return false;
    }
    stm.wait();
    return true;
}

#endif
