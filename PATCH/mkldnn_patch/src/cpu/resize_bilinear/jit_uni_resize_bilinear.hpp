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

#ifndef CPU_JIT_UNI_RESIZE_BILINEAR_HPP
#define CPU_JIT_UNI_RESIZE_BILINEAR_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_resize_bilinear_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_uni_bilinear_kernel_f32.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_resize_bilinear_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_resize_bilinear_fwd_pd_t {
        pd_t(engine_t *engine, const resize_bilinear_desc_t *adesc,
                const primitive_attr_t *attr,
                const resize_bilinear_fwd_pd_t *hint_fwd_pd)
            : cpu_resize_bilinear_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_resize_bilinear_fwd_t<isa>);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace utils;
            assert(engine()->kind() == engine_kind::cpu);
            bool ok = true
                && implication(desc()->src_desc.ndims == 5,
                        isa == avx512_common)
                && mayiuse(isa)
                && set_default_params() == status::success
                && one_of(desc()->prop_kind, forward_training,
                        forward_inference)
                && everyone_is(data_type::f32, src_pd()->desc()->data_type,
                        dst_pd()->desc()->data_type)
                && everyone_is(desired_fmt(), src_pd()->desc()->format,
                        dst_pd()->desc()->format)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (is_training) {
                auto indices_desc = *dst_pd()->desc();
                indices_desc.data_type = resize_bilinear_index_data_type(desc());
                ws_pd_ = cpu_memory_t::pd_t(engine_, &indices_desc);
            }

            return jit_uni_bilinear_kernel_f32<isa>::init_conf(jpp_, desc_,
                    src_pd_.desc(), dst_pd_.desc());
        }
        inline memory_format_t desired_fmt()
        {
            using namespace memory_format;
            return (desc()->src_desc.ndims == 4)
                ? isa == avx512_common ? nChw16c : nChw8c
                : nCdhw16c;
        }

        jit_bilinear_conf_t jpp_;

    protected:
        virtual status_t set_default_params() override {
            if (dst_pd_.desc()->format == memory_format::any)
               CHECK(dst_pd_.set_format(desired_fmt()));
            return status::success;
        }
    };

    jit_uni_resize_bilinear_fwd_t(const pd_t *pd, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*pd)
    { kernel_ = new jit_uni_bilinear_kernel_f32<isa>(conf_.jpp_); }

    ~jit_uni_resize_bilinear_fwd_t() { delete kernel_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward();
    pd_t conf_;
    jit_uni_bilinear_kernel_f32<isa> *kernel_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
