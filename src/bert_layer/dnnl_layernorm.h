// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_LAYERNORM__
#define __DNNL_LAYERNORM__

#include "dnnl_common.h"

#include <string>
#include <iostream>
#include <sstream>
#include <type_traits>


template <typename T_input, typename T_gamma, typename T_beta>
bool LayerNorm_with_gamma_beta(dnnl::engine eng, dnnl::stream stm, T_input* input, T_gamma* gamma, T_beta* beta, int m, int n) {
    char type_input = (std::is_floating_point<T_input>::value) ? 'f' : 'b';
    char type_gamma = (std::is_floating_point<T_gamma>::value) ? 'f' : 'b';
    char type_beta = (std::is_floating_point<T_beta>::value) ? 'f' : 'b';

    const void *address = static_cast<const void*>(gamma);

    std::stringstream gamma_addr;
    gamma_addr << "LayerNorm_with_gamma_beta-" << type_input << type_gamma << type_beta \
                 << '-' << m << '-' << n << '-' << address;
    std::string prim_key = gamma_addr.str();

    dnnl::memory::dims src_tz = {m, n};
    dnnl::memory::dims gamma_tz = {1, n};
    dnnl::memory::dims beta_tz = {1, n};

    dnnl::memory::data_type src_dt = dnnl::memory::data_type::f32;
    dnnl::memory::data_type gamma_dt = dnnl::memory::data_type::f32;
    dnnl::memory::data_type beta_dt = dnnl::memory::data_type::f32;

    auto it_prim_created = g_prim.find(prim_key);
    if (it_prim_created == g_prim.end()) {
        auto src_md     = dnnl::memory::desc({ src_tz }, src_dt, dnnl::memory::format_tag::nc);

        auto bnorm_d = dnnl::layer_normalization_forward::desc(
                dnnl::prop_kind::forward_inference, src_md, 9.999999960041972e-13,
                dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift );

        // Create primitive descriptor.
        auto *prim_desc = new dnnl::layer_normalization_forward::primitive_desc(bnorm_d, eng);
        auto *prim = new dnnl::layer_normalization_forward(*prim_desc);

        g_prim.insert(std::pair<std::string, dnnl::primitive *>(prim_key, prim));
        g_ln_prim_desc.insert(std::pair<std::string, dnnl::layer_normalization_forward::primitive_desc *>(prim_key, prim_desc));
    }

    auto user_src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);
    auto user_gamma_md = dnnl::memory::desc(gamma_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc); // ab or ba
    auto user_beta_md = dnnl::memory::desc(beta_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc); 

    auto user_src_memory = dnnl::memory(user_src_md, eng, input);
    auto user_gamma_memory = dnnl::memory(user_gamma_md, eng, gamma);
    auto user_beta_memory = dnnl::memory(user_beta_md, eng, beta);

    auto it_prim_desc_created = g_ln_prim_desc.find(prim_key);
    if (it_prim_desc_created == g_ln_prim_desc.end()) {
        std::cout << "batchnorm error: can find g_ln_prim_desc = " << prim_key << std::endl;
        return false;
    }
    dnnl::layer_normalization_forward::primitive_desc prim_desc = *it_prim_desc_created->second;

    auto src_memory = user_src_memory;
    auto gamma_memory = &user_gamma_memory;
    auto beta_memory = &user_beta_memory;

    if (prim_desc.src_desc() != user_src_memory.get_desc()) {
        src_memory = dnnl::memory(prim_desc.src_desc(), eng);
        auto reorder_src = dnnl::reorder(user_src_memory, src_memory);
        reorder_src.execute(stm, {
            { DNNL_ARG_FROM, user_src_memory },
            { DNNL_ARG_TO, src_memory } });
    }

    it_prim_created = g_prim.find(prim_key);
    if (it_prim_created != g_prim.end()) {
        it_prim_created->second->execute(stm, {
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
