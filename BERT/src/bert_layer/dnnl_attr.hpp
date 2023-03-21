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
            attrs_.Attr()->set_output_scales(0, {scale});
        }
        return *this;
    }

    BuildAttrs& Scale(int mask, std::vector<float> scale) {
        attrs_.Attr()->set_output_scales(mask, scale);
        return *this;
    }

    BuildAttrs& ZeroPoint(int shift, int arg = DNNL_ARG_DST) {
        if (shift != noShift) {
            attrs_.Attr()->set_zero_points(arg, 0, {shift});
        }
        return *this;
    }

    BuildAttrs& ZeroPoint(float shift, int arg = DNNL_ARG_DST) {
        return ZeroPoint(static_cast<int>(std::round(shift)), arg);
    }

    BuildAttrs& ScratchpadModeUser() {
        attrs_.Attr()->set_scratchpad_mode(dnnl::scratchpad_mode::user);
        return *this;
    }

    BuildAttrs& ScratchpadModeLib() {
        attrs_.Attr()->set_scratchpad_mode(dnnl::scratchpad_mode::library);
        return *this;
    }

    BuildAttrs& Eltwise(dnnl::algorithm algo, float alpha = 0, float beta = 0, float scale = 1.f) {
        attrs_.PostOps()->append_eltwise(scale, algo, alpha, beta);
        return *this;
    }

    BuildAttrs& Sum(float scale = 1.f) {
        attrs_.PostOps()->append_sum(scale);
        return *this;
    }

    BuildAttrs& Binary(dnnl::algorithm algo, dnnl::memory::desc memory_desc) {
        attrs_.PostOps()->append_binary(algo, memory_desc);
        return *this;
    }

    bool Empty(){
        return attrs_.Empty();
    }

    operator dnnl::primitive_attr() const { return attrs_.MakeAttr(); }

private:
    class AttrStore {
    public:
        dnnl::primitive_attr* Attr() {
            if (!attr_) {
                attr_  = dnnl::primitive_attr{};
            }
            return &attr_;
        }

        dnnl::post_ops* PostOps() {
            if (!post_ops_) {
                post_ops_ = dnnl::post_ops{};
            }
            return &post_ops_;
        }

        bool Empty() const {
            return !attr_ && !post_ops_;
        }

        dnnl::primitive_attr MakeAttr() const {
            auto result = attr_ ? attr_ : dnnl::primitive_attr{};
            if (post_ops_) {
                result.set_post_ops(post_ops_);
            }
            return result;
        }

    private:
        dnnl::primitive_attr attr_{nullptr};
        dnnl::post_ops post_ops_{nullptr};
    };

    AttrStore attrs_;
};

} // namespace dnnl_wrappers

#endif //__DNNL_ATTR__
