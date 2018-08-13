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
#ifndef PATCH_VERBOSE_HPP
#define PATCH_VERBOSE_HPP

#include "mkldnn_debug.h"
#include "c_types_map.hpp"
#include "utils.hpp"
#include "z_magic.hpp"

namespace mkldnn {
namespace impl {

#if !defined(DISABLE_VERBOSE)
#include <stdio.h>

#define PATCH_MKLDNN_VERBOSE_BUF_LEN 1024

#define PATCH_MKLDNN_VERBOSE_DAT_LEN 64
#define MKLDNN_VERBOSE_PRB_LEN 384

#define PATCH_DECL_DAT_AUX_PRB_STRS() \
    char dat_str[MKLDNN_VERBOSE_PRB_LEN] = {'\0'}; MAYBE_UNUSED(dat_str);

template <typename pd_t> static void init_info_extract_img_patches(pd_t *s, char *buffer) {
    PATCH_DECL_DAT_AUX_PRB_STRS();

    auto fmt_data = (s->desc()->prop_kind == prop_kind::backward_data
            ? s->diff_src_pd() : s->src_pd())->desc()->format;
    auto fmt_ws = s->workspace_pd()
        ? s->workspace_pd()->desc()->format : memory_format::undef;
    snprintf(dat_str, PATCH_MKLDNN_VERBOSE_DAT_LEN, "fdata:%s fws:%s",
            mkldnn_fmt2str(fmt_data), mkldnn_fmt2str(fmt_ws));

}

#else /* !defined(DISABLE_VERBOSE) */
#define PATCH_MKLDNN_VERBOSE_BUF_LEN 1

#endif /* !defined(DISABLE_VERBOSE) */

}
}

#endif
