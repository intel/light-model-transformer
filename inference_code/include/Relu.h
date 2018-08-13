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
#ifndef __RELU_H
#define __RELU_H
#include <iostream>
#include "mkldnn.hpp"

using namespace mkldnn;

memory *relu(engine *engine, convolution_forward::primitive_desc *p_prim_desc, 
                memory *p_dst_memory, float alpha, float beta, std::vector<primitive> &net) {
    auto _desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu,
            p_prim_desc->dst_primitive_desc().desc(),
            alpha, beta);
    auto _prim_desc = eltwise_forward::primitive_desc(_desc, *engine);

    memory *_dst_memory = new memory(_prim_desc.dst_primitive_desc());
    primitive *_fd = new eltwise_forward(_prim_desc, *p_dst_memory, *_dst_memory);
    net.push_back(*_fd);

    return _dst_memory;
}


memory *relu(engine *engine, deconvolution_forward::primitive_desc *p_prim_desc,
                 memory *p_dst_memory, float alpha, float beta, std::vector<primitive> &net) {
     auto _desc = eltwise_forward::desc(prop_kind::forward,
             algorithm::eltwise_relu,
             p_prim_desc->dst_primitive_desc().desc(),
             alpha, beta);
     auto _prim_desc = eltwise_forward::primitive_desc(_desc, *engine);
 
     memory *_dst_memory = new memory(_prim_desc.dst_primitive_desc());
     primitive *_fd = new eltwise_forward(_prim_desc, *p_dst_memory, *_dst_memory);
     net.push_back(*_fd);

     return _dst_memory;
 }

memory *relu(engine *engine, batch_normalization_forward::primitive_desc *p_prim_desc, 
                memory *p_dst_memory, float alpha, float beta, std::vector<primitive> &net) {
    auto _desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu,
            p_prim_desc->dst_primitive_desc().desc(),
            alpha, beta);
    auto _prim_desc = eltwise_forward::primitive_desc(_desc, *engine);

    memory *_dst_memory = new memory(_prim_desc.dst_primitive_desc());
    primitive *_fd = new eltwise_forward(_prim_desc, *p_dst_memory, *_dst_memory);
    net.push_back(*_fd);

    return _dst_memory;
}

memory *relu(engine *engine, inner_product_forward::primitive_desc *p_prim_desc, 
                memory *p_dst_memory, float alpha, float beta, std::vector<primitive> &net) {
    auto _desc = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu,
            p_prim_desc->dst_primitive_desc().desc(),
            alpha, beta);
    auto _prim_desc = eltwise_forward::primitive_desc(_desc, *engine);

    memory *_dst_memory = new memory(_prim_desc.dst_primitive_desc());
    primitive *_fd = new eltwise_forward(_prim_desc, *p_dst_memory, *_dst_memory);
    net.push_back(*_fd);

    return _dst_memory;
}
#endif
