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

#ifndef PATCH_MKLDNN_HPP
#define PATCH_MKLDNN_HPP

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stdlib.h>
#include <memory>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>

#include "mkldnn.h"
#include "mkldnn.hpp"
#include "patch_mkldnn.h"
#endif

namespace mkldnn {

/// Base class for all computational primitives.
class patch_primitive: public primitive {
public:
    /// A proxy to C primitive kind enum
    enum class kind {
        extract_image_patches = mkldnn_extract_image_patches,
        resize_bilinear = mkldnn_resize_bilinear,
    };
};

enum patch_query {
    extract_image_patches_d = mkldnn_query_extract_image_patches_d,
    resize_bilinear_d = mkldnn_query_resize_bilinear_d,
};

/// @addtogroup cpp_api_memory Memory
/// A primitive to describe data.
///
/// @addtogroup cpp_api_extract_image_patches extract_image_patches
/// A primitive to perform extract_image_patches.
///
/// @sa @ref c_api_extract_image_patches in @ref c_api
/// @{

struct extract_image_patches_forward : public patch_primitive {
    struct desc {
        mkldnn_extract_image_patches_desc_t data;
        desc(prop_kind aprop_kind, 
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const memory::dims strides,
                const memory::dims kernel,
                const memory::dims padding_l,
                const memory::dims padding_r,
                const int rate_h,
                const int rate_w,
                const padding_kind apadding_kind) {
            memory::validate_dims(strides);
            memory::validate_dims(kernel);
            memory::validate_dims(padding_l);
            memory::validate_dims(padding_r);
            error::wrap_c_api(mkldnn_extract_image_patches_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        &src_desc.data, &dst_desc.data,
                        &strides[0], &kernel[0],
                        &padding_l[0], &padding_r[0],
                        &rate_h, &rate_w,
                        mkldnn::convert_to_c(apadding_kind)),
                    "could not init a forward extract_image_patches descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
        mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(), nullptr),
                    "could not create a forward extract_image_patches primitive descriptor");
            reset(result);
        }

        memory::primitive_desc workspace_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(workspace_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a workspace primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    extract_image_patches_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(), nullptr };
        check_num_parameters(aprimitive_desc.get(), 1, 1, "extract_image_patches forward");
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a extract_image_patches forward primitive");
        reset(result);
    }

    extract_image_patches_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst, const memory &workspace) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(), workspace.get() };
        check_num_parameters(aprimitive_desc.get(), 1, 2, "extract_image_patches forward");
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a extract_image_patches forward primitive");
        reset(result);
    }
};

/// @}
/// @addtogroup cpp_api_memory Memory
/// A primitive to describe data.
///
/// @addtogroup cpp_api_resize_bilinear resize_bilinear
/// A primitive to perform resize_bilinear.
///
/// @sa @ref c_api_resize_bilinear in @ref c_api
/// @{

struct resize_bilinear_forward : public patch_primitive {
    struct desc {
        mkldnn_resize_bilinear_desc_t data;
        desc(prop_kind aprop_kind, 
                const memory::desc &src_desc,
                const memory::desc &dst_desc,
                const int align_corners) {
            error::wrap_c_api(mkldnn_resize_bilinear_forward_desc_init(&data,
                        mkldnn::convert_to_c(aprop_kind),
                        &src_desc.data, &dst_desc.data,
                        &align_corners),
                    "could not init a forward resize_bilinear descriptor");
        }
    };

    struct primitive_desc : public handle<mkldnn_primitive_desc_t> {
        primitive_desc(const desc &adesc, const engine &aengine) {
        mkldnn_primitive_desc_t result;
            error::wrap_c_api(mkldnn_primitive_desc_create(
                        &result, &adesc.data, aengine.get(), nullptr),
                    "could not create a forward resize_bilinear primitive descriptor");
            reset(result);
        }

        memory::primitive_desc workspace_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(workspace_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a workspace primititve descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc dst_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(dst_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a dst primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        memory::primitive_desc src_primitive_desc() const {
            memory::primitive_desc adesc;
            mkldnn_primitive_desc_t cdesc;
            const_mkldnn_primitive_desc_t const_cdesc =
                mkldnn_primitive_desc_query_pd(get(),
                               mkldnn::convert_to_c(src_pd), 0);
            error::wrap_c_api(mkldnn_primitive_desc_clone(&cdesc, const_cdesc),
                    "could not clone a src primitive descriptor");
            adesc.reset(cdesc);
            return adesc;
        }

        engine get_engine() { return engine::query(*this); }
    };

    resize_bilinear_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(), nullptr };
        check_num_parameters(aprimitive_desc.get(), 1, 1, "resize_bilinear forward");
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a resize_bilinear forward primitive");
        reset(result);
    }

    resize_bilinear_forward(const primitive_desc &aprimitive_desc, const primitive::at &src,
            const memory &dst, const memory &workspace) {
        mkldnn_primitive_t result;
        mkldnn_primitive_at_t inputs[] = { src.data };
        const_mkldnn_primitive_t outputs[] = { dst.get(), workspace.get() };
        check_num_parameters(aprimitive_desc.get(), 1, 2, "resize_bilinear forward");
        error::wrap_c_api(mkldnn_primitive_create(&result,
                    aprimitive_desc.get(), inputs, outputs),
                "could not create a resize_bilinear forward primitive");
        reset(result);
    }
};

/// @}

/// @} C++ API

} // namespace mkldnn

#endif
