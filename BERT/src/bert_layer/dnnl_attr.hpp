// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_ATTR__
#define __DNNL_ATTR__

#include <cmath>

#include "dnnl_common.h"

namespace dnnl_wrappers {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Attributes

class BuildAttrs {
public:
    static constexpr float noScale = 1.f;
    static constexpr int noShift = 0;

    BuildAttrs& Scale(float scale) {
        if (scale != noScale) {
            attr_.set_output_scales(0, {scale});
            empty = false;
        }
        return *this;
    }

    BuildAttrs& Scale(int mask, std::vector<float> scale) {
        attr_.set_output_scales(mask, scale);
        empty = false;
        return *this;
    }

    BuildAttrs& ZeroPoint(int shift, int arg = DNNL_ARG_DST) {
        if (shift != noShift) {
            attr_.set_zero_points(arg, 0, {shift});
            empty = false;
        }
        return *this;
    }

    BuildAttrs& ZeroPoint(float shift, int arg = DNNL_ARG_DST) {
        return ZeroPoint(static_cast<int>(std::round(shift)), arg);
    }

    BuildAttrs& Eltwise(dnnl::algorithm algo, float alpha = 0, float beta = 0, float scale = 1.f) {
        post_ops_.append_eltwise(scale, algo, alpha, beta);
        empty = false;
        return *this;
    }

    BuildAttrs& Sum(float scale = 1.f) {
        post_ops_.append_sum(scale);
        empty = false;
        return *this;
    }

    BuildAttrs& Binary(dnnl::algorithm algo, dnnl::memory::desc memory_desc) {
        post_ops_.append_binary(algo, memory_desc);
        empty = false;
        return *this;
    }

    bool Empty(){
        return empty;
    }

    operator dnnl::primitive_attr() const { return MakeAttr_(); }

private:
    dnnl::primitive_attr MakeAttr_() const {
        auto result = attr_;
        result.set_post_ops(post_ops_);
        return result;
    }

    dnnl::primitive_attr attr_;
    dnnl::post_ops post_ops_;
    bool empty = true;
};

} // namespace dnnl_wrappers

#endif //__DNNL_ATTR__
