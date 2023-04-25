// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_ATTR__
#define __DNNL_ATTR__

#include <cmath>
#include <unordered_map>

#include "dnnl_common.h"

namespace dnnl_wrappers {

///////////////////////////////////////////////////////////////////////////////////////////////////
// Attributes

class BuildAttrs {
public:
    static constexpr float noScale = 1.f;
    static constexpr int32_t noShift = 0;

    BuildAttrs& Scale(const dnnl::memory& scale, int arg, int mask = 0) {
        if (scale) {
            attrs_.Attr()->set_scales_mask(arg, mask);
            attrs_.Args()[DNNL_ARG_ATTR_SCALES | arg] = scale;
        }
        return *this;
    }

    BuildAttrs& ZeroPoint(const dnnl::memory& shift, int arg, int mask = 0) {
        if (shift) {
            attrs_.Attr()->set_zero_points_mask(arg, mask);
            attrs_.Args()[DNNL_ARG_ATTR_ZERO_POINTS | arg] = shift;
        }
        return *this;
    }

    BuildAttrs& ScratchpadModeUser() {
        attrs_.Attr()->set_scratchpad_mode(dnnl::scratchpad_mode::user);
        return *this;
    }

    BuildAttrs& ScratchpadModeLib() {
        attrs_.Attr()->set_scratchpad_mode(dnnl::scratchpad_mode::library);
        return *this;
    }

    BuildAttrs& Eltwise(dnnl::algorithm algo, float alpha = 0, float beta = 0) {
        attrs_.PostOps()->append_eltwise(algo, alpha, beta);
        return *this;
    }

    BuildAttrs& Sum(float scale = 1.f) {
        attrs_.PostOps()->append_sum(scale);
        return *this;
    }

    BuildAttrs& Binary(dnnl::algorithm algo, const dnnl::memory::desc& memory_desc) {
        attrs_.PostOps()->append_binary(algo, memory_desc);
        return *this;
    }

    bool Empty(){
        return attrs_.Empty();
    }

    dnnl::primitive_attr GetAttrs() const { return attrs_.MakeAttr(); }

    const std::unordered_map<int, dnnl::memory>& GetArgs() const { return attrs_.Args(); }

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

        std::unordered_map<int, dnnl::memory>& Args() {
            return args_;
        }

        const std::unordered_map<int, dnnl::memory>& Args() const {
            return args_;
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
        std::unordered_map<int, dnnl::memory> args_;
    };

    AttrStore attrs_;
};

} // namespace dnnl_wrappers

#endif //__DNNL_ATTR__
