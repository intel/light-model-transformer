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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "cpu_extract_image_patches_pd.hpp"

#include "jit_uni_extract_img_patches_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
//using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_extract_img_patches_call_s, field)

template <cpu_isa_t isa>
status_t jit_uni_extract_img_patches_kernel_f32<isa>::init_conf(jit_extract_img_patches_conf_t &jpp,
            const extract_image_patches_desc_t &pd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &dst_d) {

    const int simd_w = isa == avx512_common ? 16 : 8;
    const int ndims = src_d.ndims();

    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];

    jpp.c = utils::rnd_up(src_d.dims()[1], simd_w);
    if (jpp.c > src_d.blocking_desc().padding_dims[1])
        return status::unimplemented;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = src_d.dims()[ndims-2];
    jpp.iw = src_d.dims()[ndims-1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.oh = dst_d.dims()[ndims-2];
    jpp.ow = dst_d.dims()[ndims-1];

    jpp.stride_d = (ndims == 5 ) ? pd.strides[0] : 1;
    jpp.stride_h = pd.strides[ndims-4];
    jpp.stride_w = pd.strides[ndims-3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = pd.kernel[ndims-4];
    jpp.kw = pd.kernel[ndims-3];

    jpp.f_pad = (ndims == 5 ) ? pd.padding[0][0] : 0;
    jpp.t_pad = pd.padding[0][ndims-4];
    jpp.l_pad = pd.padding[0][ndims-3];

    jpp.rate_h = pd.rate_h;
    jpp.rate_w = pd.rate_w;

    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
//    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;
    jpp.ind_dt = extract_image_patches_index_data_type(&pd);

#if 0
    jpp.simple_alg = jpp.is_training
        || utils::implication(jpp.is_backward, jpp.kd <= jpp.stride_d);
#else
    jpp.simple_alg = jpp.is_training;
#endif

    jpp.c_block = simd_w;

    jpp.nb_c = jpp.c / jpp.c_block;

    jpp.ur_w = isa == avx512_common ? 16 : 4;
    if (jpp.is_training)
        jpp.ur_w = isa == avx512_common ? 9 : 3;
#if 0
    else if (jpp.is_backward)
        jpp.ur_w = isa == avx512_common ? 6 : 3;
#endif

    if (jpp.ow < jpp.ur_w) jpp.ur_w = jpp.ow;
    if (jpp.l_pad > jpp.ur_w) return status::unimplemented;

    jpp.ur_w_tail = jpp.ow % jpp.ur_w;

    return status::success;
}

template struct jit_uni_extract_img_patches_kernel_f32<sse42>;
template struct jit_uni_extract_img_patches_kernel_f32<avx2>;
template struct jit_uni_extract_img_patches_kernel_f32<avx512_common>;
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
