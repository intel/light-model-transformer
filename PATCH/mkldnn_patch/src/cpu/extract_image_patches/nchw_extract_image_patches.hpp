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

#ifndef CPU_NCHW_EXTRACT_IMAGE_PATCHES_HPP
#define CPU_NCHW_EXTRACT_IMAGE_PATCHES_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_extract_image_patches_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;

template <impl::data_type_t data_type>
struct nchw_extract_image_patches_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_extract_image_patches_fwd_pd_t {
        pd_t(engine_t *engine, const extract_image_patches_desc_t *adesc,
                const primitive_attr_t *attr,
                const extract_image_patches_fwd_pd_t *hint_fwd_pd)
            : cpu_extract_image_patches_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("nchw_extract_image_patches:any", nchw_extract_image_patches_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            assert(engine()->kind() == engine_kind::cpu);
            auto src_format = src_pd()->desc()->format;
            bool ok = true
                && set_default_params() == status::success
                && utils::one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && utils::everyone_is(data_type, src_pd()->desc()->data_type,
                        dst_pd()->desc()->data_type)
                && utils::one_of(src_format, nchw, ncdhw)
                && (src_format == dst_pd()->desc()->format)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (is_training) {
                auto indices_desc = *dst_pd()->desc();
                indices_desc.data_type = extract_image_patches_index_data_type(desc());
                ws_pd_ = cpu_memory_t::pd_t(engine_, &indices_desc);
            }

            return status::success;
        }
    };

    nchw_extract_image_patches_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
