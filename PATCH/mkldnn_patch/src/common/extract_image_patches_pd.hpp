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
/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
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

#ifndef extract_image_patches_PD_HPP
#define extract_image_patches_PD_HPP

#include "mkldnn.h"
#include "patch_mkldnn.hpp"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "memory_pd.hpp"

namespace mkldnn {
namespace impl {

struct extract_image_patches_fwd_pd_t: public primitive_desc_t {
    typedef extract_image_patches_fwd_pd_t base_class;
    typedef extract_image_patches_fwd_pd_t hint_class;
    static constexpr auto base_pkind = primitive_kind::extract_image_patches;

    extract_image_patches_fwd_pd_t(mkldnn::impl::engine_t *engine,
            const extract_image_patches_desc_t *adesc, const primitive_attr_t *attr,
            const extract_image_patches_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, primitive_kind::extract_image_patches)
        , desc_(*adesc), hint_fwd_pd_(hint_fwd_pd) {}
    virtual ~extract_image_patches_fwd_pd_t() {}

    const extract_image_patches_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { init_info_extract_img_patches(this, this->info_); }

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
        case query::extract_image_patches_d:
            *(const extract_image_patches_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common extract_image_patches aux functions */
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
    inline int KD() const { return is_3d() ? desc_.kernel[0] : 1; }
    inline int KH() const
    { return is_3d() ? desc_.kernel[1] : desc_.kernel[0]; }
    inline int KW() const
    { return is_3d() ? desc_.kernel[2] : desc_.kernel[1]; }

    inline int KSD() const { return is_3d() ? desc_.strides[0] : 1; }
    inline int KSH() const
    { return is_3d() ? desc_.strides[1] : desc_.strides[0]; }
    inline int KSW() const
    { return is_3d() ? desc_.strides[2] : desc_.strides[1]; }

    inline int padFront() const { return is_3d() ? desc_.padding[0][0] : 0; }
    inline int padBack() const { return is_3d() ? desc_.padding[1][0] : 0; }
    inline int padT() const { return is_3d()
        ? desc_.padding[0][1] : desc_.padding[0][0]; }
    inline int padB() const { return is_3d()
        ? desc_.padding[1][1] : desc_.padding[1][0]; }
    inline int padL() const { return is_3d()
        ? desc_.padding[0][2] : desc_.padding[0][1]; }
    inline int padR() const { return is_3d()
        ? desc_.padding[1][2] : desc_.padding[1][1]; }
    inline int rateH() const { return desc_.rate_h; }
    inline int rateW() const { return desc_.rate_w; }
protected:
    extract_image_patches_desc_t desc_;
    const extract_image_patches_fwd_pd_t *hint_fwd_pd_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s

