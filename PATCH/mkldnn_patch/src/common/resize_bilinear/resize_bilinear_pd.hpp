/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef RESIZE_BILINEAR_PD_HPP
#define RESIZE_BILINEAR_PD_HPP

#include "mkldnn.h"
#include "patch_mkldnn.hpp"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"

namespace mkldnn {
namespace impl {


struct CachedInterpolation {
    int lower;  // Lower source index used in the interpolation
    int upper;  // Upper source index used in the interpolation
    float lerp;
};

struct resize_bilinear_fwd_pd_t: public primitive_desc_t {
    typedef resize_bilinear_fwd_pd_t base_class;
    typedef resize_bilinear_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::resize_bilinear;

    resize_bilinear_fwd_pd_t(mkldnn::impl::engine_t *engine,
            const resize_bilinear_desc_t *adesc, const primitive_attr_t *attr,
            const resize_bilinear_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, primitive_kind::resize_bilinear)
        , desc_(*adesc), hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~resize_bilinear_fwd_pd_t() {}

    const resize_bilinear_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_bilinear(this, this->info_); }

    virtual const memory_pd_t *input_pd(int index = 0) const override
    { return index == 0 ? src_pd() : nullptr; }
    virtual const memory_pd_t *output_pd(int index = 0) const override {
        if (index == 0) return dst_pd();
        if (index == 1) return workspace_pd();
        return nullptr;
    }

    virtual int n_inputs() const override { return 1; }
    virtual int n_outputs() const override
    { return 1 + (workspace_pd() != nullptr); }

    virtual status_t query(query_t what, int idx, void *result) const override
    {
        switch (what) {
        case query::resize_bilinear_d:
            *(const resize_bilinear_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common resize_bilinear aux functions */
    inline bool is_3d() const { return desc_.src_desc.ndims == 5; }

    inline int MB() const { return desc_.src_desc.dims[0]; }
    inline int C() const { return desc_.src_desc.dims[1]; }
    inline int ID() const { return is_3d() ? desc_.src_desc.dims[2] : 1; }
    inline int IH() const { return is_3d()
        ? desc_.src_desc.dims[3] : desc_.src_desc.dims[2]; }
    inline int IW() const { return is_3d()
        ? desc_.src_desc.dims[4] : desc_.src_desc.dims[3]; }
    inline int OD() const { return is_3d()
        ? desc_.dst_desc.dims[2] : 1; }
    inline int OH() const { return is_3d()
        ? desc_.dst_desc.dims[3] : desc_.dst_desc.dims[2]; }
    inline int OW() const { return is_3d()
        ? desc_.dst_desc.dims[4] : desc_.dst_desc.dims[3]; }
    inline int alignCorners() const { return desc_.align_corners; }
    inline float RATE(const int _in, const int _out) const {
        float float_in = _in * 1.0;
        float scale = 0.0;
        if (desc_.align_corners && _out > 1)
            scale = (float_in - 1) / (_out - 1);
        else
            scale = float_in / _out;

        return scale;
    }
    inline void interpolationWeights(const int in_size, 
                                    const int out_size, 
                                    const float scale, 
                                    CachedInterpolation* interpolation) const {
        interpolation[out_size].lower = 0;
        interpolation[out_size].upper = 0;
        for (int i = out_size - 1; i >= 0; --i) {
            const float in = i * scale;
            interpolation[i].lower = static_cast<int>(in);
            interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
            interpolation[i].lerp = in - interpolation[i].lower;
        }
    }
protected:
    resize_bilinear_desc_t desc_;
    const resize_bilinear_fwd_pd_t *hint_fwd_pd_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

