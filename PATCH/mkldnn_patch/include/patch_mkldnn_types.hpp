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

#ifndef PATCH_MKLDNN_TYPES_H
#define PATCH_MKLDNN_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
#include <stddef.h>
#include <stdint.h>
#include "mkldnn_types.h"
#endif

/** A descriptor of a extract_image_patches operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_extract_image_patches. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Source gradient memory descriptor. */
    mkldnn_memory_desc_t diff_src_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_dst_desc;
    /** extract_image_patches kernel strides for spatial dimensions. */
    mkldnn_dims_t strides;
    /** extract_image_patches kernel spatial dimensions. */
    mkldnn_dims_t kernel;
    /** Padding in each spatial dimension. padding[0] is a padding in the
     * beginning (@p padding_l), padding[1] is a padding in the end (@p
     * padding_r). */
    mkldnn_dims_t padding[2];
    /** The kind of padding to use. */
    mkldnn_padding_kind_t padding_kind;
    /** The accumulator data type. Initialized automatically. */
    mkldnn_data_type_t accum_data_type;
    int rate_h;
    int rate_w;
} mkldnn_extract_image_patches_desc_t;

/** A descriptor of a resize_bilinear operation. */
typedef struct {
    /** The kind of primitive. Used for self identifying the primitive
     * descriptor. Must be #mkldnn_resize_bilinear. */
    mkldnn_primitive_kind_t primitive_kind;
    /** The kind of propagation. Possible values: #mkldnn_forward_training,
     * #mkldnn_forward_inference, #mkldnn_backward, and #mkldnn_backward_data.
     */
    mkldnn_prop_kind_t prop_kind;
    /** Source memory descriptor. */
    mkldnn_memory_desc_t src_desc;
    /** Source gradient memory descriptor. */
    mkldnn_memory_desc_t diff_src_desc;
    /** Destination memory descriptor. */
    mkldnn_memory_desc_t dst_desc;
    /** Destination gradient memory descriptor. */
    mkldnn_memory_desc_t diff_dst_desc;
    /** The accumulator data type. Initialized automatically. */
    mkldnn_data_type_t accum_data_type;
    int align_corners;
} mkldnn_resize_bilinear_desc_t;

#ifdef __cplusplus
}
#endif


#endif
