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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_uni_extract_image_patches.hpp"
#include "type_helpers.hpp"
#include "nstl.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
void jit_uni_extract_image_patches_fwd_t<isa>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto dst = reinterpret_cast<data_t*>(this->memory(0));

    const memory_desc_wrapper src_d(conf_.src_pd());
    const memory_desc_wrapper dst_d(conf_.dst_pd());
    const memory_desc_wrapper indices_d(conf_.workspace_pd());


    const auto &jpp = conf_.jpp_;

    const int KH = jpp.kh;
    const int KW = jpp.kw;
    const int SH = jpp.stride_h;
    const int SW = jpp.stride_w;

    const int MB = jpp.mb; // batch size

    const int IC = jpp.c;
    const int X = jpp.c_block; // x of nChw'x'c
    const int C = jpp.nb_c; // C of n'C'hwxc
    const int OC = SH * SW * IC;

    const int IH = jpp.ih;
    const int IW = jpp.iw;
    const int OH = jpp.oh;
    const int OW = jpp.ow;

    const int padT = jpp.t_pad;
    const int padL = jpp.l_pad;

    const int rateH = jpp.rate_h;
    const int rateW = jpp.rate_w;

    auto ker = [&](int _b, int _oh, int _ow) {
        int _oC = 0;
        long src_id = 0, dst_id = 0;
        for ( int _kh = 0; _kh < KH; _kh ++ ) {
            const int _ih = _oh * SH + _kh * rateH - padT;
            for ( int _kw = 0; _kw < KW; _kw ++ ) {
                const int _iw = _ow * SW + _kw * rateW - padL;
                for ( int _iC = 0; _iC < C; _iC ++ ) {
                    src_id = _b*IC*IH*IW + _iC*X*IH*IW + _ih*IW*X + _iw*X;
                    dst_id = _b*OC*OH*OW + _oC*X*OH*OW + _oh*OW*X + _ow*X;
                    for ( int _ix = 0, _ox = 0; _ix < X; _ix ++, _ox ++ ) {
                        if (_ih < 0 || _ih >= IH) dst[dst_id+_ox] = 0.0;
                        else if (_iw < 0 || _iw >= IW) dst[dst_id+_ox] = 0.0;
                        else dst[dst_id+_ox] = src[src_id+_ix];
                    }
                    _oC ++;
                }
            }
        }
    };

#   pragma omp parallel for collapse(3) schedule(static)
    for (int _b = 0; _b < MB; _b ++) {
        for ( int _oh = 0; _oh < jpp.oh; _oh ++ ) {
            for ( int _ow = 0; _ow < jpp.ow; _ow ++ ) {
                ker(_b, _oh, _ow);
            }
        }
    }    
}

template struct jit_uni_extract_image_patches_fwd_t<sse42>;
template struct jit_uni_extract_image_patches_fwd_t<avx2>;
template struct jit_uni_extract_image_patches_fwd_t<avx512_common>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
