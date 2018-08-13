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
}

namespace query {
    const mkldnn_query_t extract_image_patches_d = mkldnn_query_extract_image_patches_d;
}

using extract_image_patches_desc_t = mkldnn_extract_image_patches_desc_t;

#if 0
struct patch_op_desc_t {
    union {
        primitive_kind_t patch_kind;
        extract_image_patches_desc_t mkldnn_extract_image_patches;
    };

    patch_op_desc_t(const primitive_kind_t &_): patch_kind(_) {}
#   define PATCH_DECL_CTOR_AND_CONVERTERS(c_type, name) \
    patch_op_desc_t(const c_type &_): name(_) {} \
    static patch_op_desc_t *convert_from_c(c_type *_) \
    { return reinterpret_cast<patch_op_desc_t*>(_); } \
    static const patch_op_desc_t *convert_from_c(const c_type *_) \
    { return reinterpret_cast<const patch_op_desc_t*>(_); }


    PATCH_DECL_CTOR_AND_CONVERTERS(extract_image_patches_desc_t, mkldnn_extract_image_patches);
#   undef PATCH_DECL_CTOR_AND_CONVERTERS
};
#endif

}
}

#ifdef __cplusplus
}


#endif
#endif
