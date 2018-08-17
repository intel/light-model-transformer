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

#ifndef PATCH_MKLDNN_TYPES_MAP_HPP
#define PATCH_MKLDNN_TYPES_MAP_HPP

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stddef.h>
#include <stdint.h>
#include "mkldnn_types.h"
#endif


namespace mkldnn {
namespace impl {

using primitive_kind_t = mkldnn_primitive_kind_t;
namespace primitive_kind {
    const primitive_kind_t extract_image_patches = mkldnn_extract_image_patches;
    const primitive_kind_t resize_bilinear = mkldnn_resize_bilinear;
}

namespace query {
    const mkldnn_query_t extract_image_patches_d = mkldnn_query_extract_image_patches_d;
    const mkldnn_query_t resize_bilinear_d = mkldnn_query_resize_bilinear_d;
}

using extract_image_patches_desc_t = mkldnn_extract_image_patches_desc_t;
using resize_bilinear_desc_t = mkldnn_resize_bilinear_desc_t;

}
}

#ifdef __cplusplus
}


#endif
#endif
