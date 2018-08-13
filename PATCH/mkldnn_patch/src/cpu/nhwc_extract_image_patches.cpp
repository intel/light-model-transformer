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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"

#include "nhwc_extract_image_patches.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

#define MEM_D(name) name##_d

#define DECLARE_READ_STRIDES(name)                                             \
    const size_t name##_n_stride = MEM_D(name).blocking_desc().strides[0][0];  \
    const size_t name##_d_stride = (!is_3d)                                    \
                                 ? 0                                           \
                                 : MEM_D(name).blocking_desc().strides[0][2];  \
    const size_t name##_h_stride = (!is_3d)                                    \
                                 ? MEM_D(name).blocking_desc().strides[0][2]   \
                                 : MEM_D(name).blocking_desc().strides[0][3];  \
    const size_t name##_w_stride = (!is_3d)                                    \
                                 ? MEM_D(name).blocking_desc().strides[0][3]   \
                                 : MEM_D(name).blocking_desc().strides[0][4];

namespace nhwc_extract_image_patches {
    size_t strided_offset(const int _n, const size_t _sn,
                          const int _d, const size_t _sd,
                          const int _h, const size_t _sh,
                          const int _w, const size_t _sw)
    {
        return   _n * _sn
               + _d * _sd
               + _h * _sh
               + _w * _sw;
    }
}

template <impl::data_type_t data_type>
void nhwc_extract_image_patches_fwd_t<data_type>::array_div_by_const(const int n,
                   const data_t *src, const size_t num, data_t *dst)
{
    for (int i = 0; i < n; ++i)
    {
        float ftmp = (float)src[i];
        ftmp = ftmp / num;
        dst[i] = math::out_round<data_t>(ftmp);
    }
}

template <impl::data_type_t data_type>
void nhwc_extract_image_patches_fwd_t<data_type>::array_add(const int n,
                                              const data_t *src, data_t *dst)
{
    for (int i = 0;  i < n; ++i)
    {
        dst[i] += src[i];
    }
}

template <impl::data_type_t data_type>
void nhwc_extract_image_patches_fwd_t<data_type>::execute_forward() {
    using namespace prop_kind;
    using namespace nhwc_extract_image_patches;


    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t *>(this->memory(0));

    const memory_desc_wrapper MEM_D(dst)(conf_.dst_pd());
    const memory_desc_wrapper MEM_D(src)(conf_.src_pd());

    const int IH = conf_.IH();
    const int IW = conf_.IW();
    const int KH = conf_.KH();
    const int KW = conf_.KW();
    const int SH = conf_.KSH();
    const int SW = conf_.KSW();
    const int padT = conf_.padT();
    const int padL = conf_.padL();
    const int MB = conf_.MB();
    const int IC = conf_.C();
    const int OC = conf_.C() * KH * KW;
    const int OH = conf_.OH();
    const int OW = conf_.OW();
    const int rateH = conf_.rateH();
    const int rateW = conf_.rateW();

    auto ker = [&](int _b, int _oh, int _ow) {
        long src_batch_id = _b * IH * IW * IC;
        long dst_batch_id = _b * OH * OW * OC;

        int _oc = 0;
        long src_id = 0, dst_id = 0;
        for ( int _kh = 0; _kh < KH; _kh ++ ) {
            const int _ih = _oh * SH + _kh * rateH - padT;
            for ( int _kw = 0; _kw < KW; _kw ++ ) {
                const int _iw = _ow * SW + _kw * rateW - padL;
                for ( int _ic = 0; _ic < IC; _ic ++, _oc ++ ) {
                    src_id = src_batch_id + _ih*IW*IC + _iw*IC + _ic;
                    dst_id = dst_batch_id + _oh*OW*OC + _ow*OC + _oc;

                    if (_ih < 0 || _ih >= IH) dst[dst_id] = 0.0;
                    else if (_iw < 0 || _iw >= IW) dst[dst_id] = 0.0;
                    else dst[dst_id] = src[src_id];
                }
            }
        }
    };

#   pragma omp parallel for collapse(3) schedule(static)
    for (int _b = 0; _b < MB; _b ++) {
        for ( int _oh = 0; _oh < OH; _oh ++ ) {
            for ( int _ow = 0; _ow < OW; _ow ++ ) {
                ker(_b, _oh, _ow);
            }
        }
    }    
}

template struct nhwc_extract_image_patches_fwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
