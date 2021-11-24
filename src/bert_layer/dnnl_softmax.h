// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_SOFTMAX__
#define __DNNL_SOFTMAX__

#include "dnnl_common.h"

template <typename T_input>
bool Softmax(engine eng, stream stm, T_input* input, int m, int n) {
    memory::dims src_tz = {m, n};
    memory::data_type src_dt = memory::data_type::f32;

    auto src_md = memory::desc(src_tz, src_dt, memory::format_tag::nc);

    const int axis = 1;
    auto softmax_d = softmax_forward::desc(prop_kind::forward_inference, src_md, axis);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_d, eng);
    auto softmax_prim = softmax_forward(softmax_pd);

    auto user_src_md = memory::desc(src_tz, memory::data_type::f32, memory::format_tag::nc);
    auto user_src_memory = memory(user_src_md, eng, input);

    auto src_memory = user_src_memory;

    if (softmax_pd.src_desc() != user_src_memory.get_desc()) {
        src_memory = memory(softmax_pd.src_desc(), eng);
        auto reorder_src = reorder(user_src_memory, src_memory);
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
