// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_SOFTMAX__
#define __DNNL_SOFTMAX__

#include "dnnl_common.h"


template <typename T_input>
bool Softmax(dnnl::engine eng, dnnl::stream stm, T_input* input, int m, int n) {
    dnnl::memory::dims src_tz = {m, n};
    dnnl::memory::data_type src_dt = dnnl::memory::data_type::f32;

    auto src_md = dnnl::memory::desc(src_tz, src_dt, dnnl::memory::format_tag::nc);

    const int axis = 1;
    auto softmax_d = dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference, src_md, axis);
    auto softmax_pd = dnnl::softmax_forward::primitive_desc(softmax_d, eng);
    auto softmax_prim = dnnl::softmax_forward(softmax_pd);

    auto user_src_md = dnnl::memory::desc(src_tz, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);
    auto user_src_memory = dnnl::memory(user_src_md, eng, input);

    auto src_memory = user_src_memory;

    if (softmax_pd.src_desc() != user_src_memory.get_desc()) {
        src_memory = dnnl::memory(softmax_pd.src_desc(), eng);
        auto reorder_src = dnnl::reorder(user_src_memory, src_memory);
        reorder_src.execute(stm, {
            { DNNL_ARG_FROM, user_src_memory },
            { DNNL_ARG_TO, src_memory } });
    }

    softmax_prim.execute(stm, {
        { DNNL_ARG_SRC, src_memory },
        { DNNL_ARG_DST, src_memory } });
    
    stm.wait();
    return true;
}

#endif
