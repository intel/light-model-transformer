// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_OPS__
#define __DNNL_OPS__

#include "dnnl_common.h"
#include "dnnl_attr.hpp"
#include "dnnl_data.hpp"

#include <string>

namespace dnnl_wrappers {

///////////////////////////////////////////////////////////////////////////////////////////////////
// MatMul

class MatMul {
public:
    MatMul(const dnnl::engine& eng,
        const dnnl::memory::desc& src_md, const dnnl::memory::desc& weights_md,
        const dnnl::memory::desc& bias_md, const dnnl::memory::desc& dst_md,
        const dnnl::primitive_attr& attr)
        : prim_{dnnl::matmul::primitive_desc{dnnl::matmul::desc{src_md, weights_md, bias_md, dst_md}, attr, eng}} {
    }

    MatMul(const dnnl::primitive& prim) : prim_(prim) {}

    void Compute(dnnl::stream& stm, DataSource& src, DataSource& weights, DataSource& bias, dnnl::memory& dst_memory) {
        const auto prim_desc = PrimDesc();
        assert(prim_desc.dst_desc() == dst_memory.get_desc());

        auto src_memory = src.GetData(stm, prim_desc.src_desc());
        auto weights_memory = weights.GetData(stm, prim_desc.weights_desc());
        auto bias_memory = bias.GetData(stm, prim_desc.bias_desc());
        prim_.execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_WEIGHTS, weights_memory },
            { DNNL_ARG_BIAS, bias_memory },
            { DNNL_ARG_DST, dst_memory } });
        // FIXME(rfsaliev) have to wait due to lifetime of x_memory variables
        stm.wait();
    }

    void ComputeWithPostOps(dnnl::stream& stm, DataSource& src, DataSource& weights,
                            std::unordered_map<int, std::reference_wrapper<DataSource>>& post_op_data,
                            dnnl::memory& dst_memory) {
        const auto prim_desc = PrimDesc();
        assert(prim_desc.dst_desc() == dst_memory.get_desc());

        auto src_memory = src.GetData(stm, prim_desc.src_desc());
        auto weights_memory = weights.GetData(stm, prim_desc.weights_desc());

        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, src_memory},
            {DNNL_ARG_WEIGHTS, weights_memory},
            {DNNL_ARG_DST, dst_memory}
        };

        // (krzychut)
        // Due to https://github.com/oneapi-src/oneDNN/issues/1337,
        // we have to get the attrs and post_ops in separate lines to ensure
        // proper lifetime of the attr object.
        auto attr = prim_desc.get_primitive_attr();
        auto post_ops = attr.get_post_ops();
        // Do NOT chain get_primitive_attr().get_post_ops() like this until the above issue is fixed:
        // auto post_ops = prim_desc.get_primitive_attr().get_post_ops(); 

        for(auto& item : post_op_data)
        {
            auto idx = item.first;
            auto& data_source = item.second.get();
            dnnl::algorithm alg;
            dnnl::memory::desc desc;
            post_ops.get_params_binary(idx, alg, desc);
            auto data = data_source.GetData(stm, desc);
            args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1, data);
        }

        prim_.execute(stm, args);
        // FIXME(rfsaliev) have to wait due to lifetime of x_memory variables
        stm.wait();
    }

    dnnl::matmul::primitive_desc PrimDesc() const {
        auto c_desc = prim_.get_primitive_desc();
        return dnnl::matmul::primitive_desc{const_cast<dnnl_primitive_desc_t>(c_desc)};
    }

    const dnnl::primitive& Prim() const {
        return prim_;
    }

private:
    dnnl::primitive prim_;
};

struct MatMulDims {
    MatMulDims(int batch, int m, int n, int k, bool bias_2d = false)
        : src_tz{batch, m, k}
        , weights_tz{1, k, n}
        , bias_tz{1, bias_2d ? m : 1, n}
        , dst_tz{batch, m, n} {}

    dnnl::memory::dims src_tz;
    dnnl::memory::dims weights_tz;
    dnnl::memory::dims bias_tz;
    dnnl::memory::dims dst_tz;
};

inline MatMul MakeMatMul(const dnnl::engine& eng, int batch, int m, int n, int k, 
                        dnnl::memory::data_type src_dt, dnnl::memory::data_type weights_dt,
                        dnnl::memory::data_type bias_dt, dnnl::memory::data_type dst_dt,
                        const dnnl::primitive_attr& attr = {}) {

    const MatMulDims dims{batch, m, n, k};

    // plain memory format can be defined using empty strides argument
    dnnl::memory::dims plain{};
    using fmt = dnnl::memory::format_tag;

    const auto src_md     = dnnl::memory::desc(dims.src_tz , src_dt, plain);
    const auto weights_md = dnnl::memory::desc(dims.weights_tz, weights_dt, fmt::any);
    const auto bias_md    = dnnl::memory::desc(dims.bias_tz, bias_dt, plain);
    const auto dst_md     = dnnl::memory::desc(dims.dst_tz, dst_dt, plain);

    return MatMul{eng, src_md, weights_md, bias_md, dst_md, attr};
}

template <typename T_src, typename T_wei, typename T_bias, typename T_dst>
MatMul MakeMatMul(const dnnl::engine& eng, int batch, int m, int n, int k, const dnnl::primitive_attr& attr = {}) {
    const auto src_dt = DnnlDataType<T_src>::value;
    const auto weights_dt = DnnlDataType<T_wei>::value;
    const auto bias_dt = DnnlDataType<T_bias>::value;
    const auto dst_dt = DnnlDataType<T_dst>::value;
    return MakeMatMul(eng, batch, m, n, k, src_dt, weights_dt, bias_dt, dst_dt, attr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// InnerProduct
class InnerProduct {
public:
    InnerProduct(const dnnl::engine& eng,
        const dnnl::memory::desc& src_md, const dnnl::memory::desc& weights_md,
        const dnnl::memory::desc& bias_md, const dnnl::memory::desc& dst_md,
        const dnnl::primitive_attr& attr)
        : prim_{dnnl::inner_product_forward::primitive_desc{
            dnnl::inner_product_forward::desc{dnnl::prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md},
            attr,
            eng}} {
    }

    InnerProduct(const dnnl::primitive& prim) : prim_(prim) {}

    void Compute(dnnl::stream& stm, DataSource& src, DataSource& weights, DataSource& bias, dnnl::memory& dst_memory) {
        const auto prim_desc = PrimDesc();
        assert(prim_desc.dst_desc() == dst_memory.get_desc());

        auto src_memory = src.GetData(stm, prim_desc.src_desc());
        auto weights_memory = weights.GetData(stm, prim_desc.weights_desc());
        auto bias_memory = bias.GetData(stm, prim_desc.bias_desc());
        prim_.execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_WEIGHTS, weights_memory },
            { DNNL_ARG_BIAS, bias_memory },
            { DNNL_ARG_DST, dst_memory } });
        // FIXME(rfsaliev) have to wait due to lifetime of x_memory variables
        stm.wait();
    }

    void ComputeWithPostOps(dnnl::stream& stm, DataSource& src, DataSource& weights,
                            std::unordered_map<int, std::reference_wrapper<DataSource>>& post_op_data,
                            dnnl::memory& dst_memory) {
        const auto prim_desc = PrimDesc();
        assert(prim_desc.dst_desc() == dst_memory.get_desc());

        auto src_memory = src.GetData(stm, prim_desc.src_desc());
        auto weights_memory = weights.GetData(stm, prim_desc.weights_desc());

        std::unordered_map<int, dnnl::memory> args = {
            {DNNL_ARG_SRC, src_memory},
            {DNNL_ARG_WEIGHTS, weights_memory},
            {DNNL_ARG_DST, dst_memory}
        };

        // (krzychut)
        // Due to https://github.com/oneapi-src/oneDNN/issues/1337,
        // we have to get the attrs and post_ops in separate lines to ensure
        // proper lifetime of the attr object.
        auto attr = prim_desc.get_primitive_attr();
        auto post_ops = attr.get_post_ops();
        // Do NOT chain get_primitive_attr().get_post_ops() like this until the above issue is fixed:
        // auto post_ops = prim_desc.get_primitive_attr().get_post_ops(); 

        for(auto& item : post_op_data)
        {
            auto idx = item.first;
            auto& data_source = item.second.get();
            dnnl::algorithm alg;
            dnnl::memory::desc desc;
            post_ops.get_params_binary(idx, alg, desc);
            auto data = data_source.GetData(stm, desc);
            args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1, data);
        }

        prim_.execute(stm, args);
        // FIXME(rfsaliev) have to wait due to lifetime of x_memory variables
        stm.wait();
    }

    dnnl::inner_product_forward::primitive_desc PrimDesc() const {
        auto c_desc = prim_.get_primitive_desc();
        return dnnl::inner_product_forward::primitive_desc{const_cast<dnnl_primitive_desc_t>(c_desc)};
    }

    const dnnl::primitive& Prim() const {
        return prim_;
    }

private:
    dnnl::primitive prim_;
};

struct InnerProductDims {
    InnerProductDims(int batch, int m, int n, int k)
        : src_tz{batch * m, k}
        , weights_tz{n, k}
        , bias_tz{n}
        , dst_tz{batch * m, n} {}

    dnnl::memory::dims src_tz;
    dnnl::memory::dims weights_tz;
    dnnl::memory::dims bias_tz;
    dnnl::memory::dims dst_tz;
};

inline InnerProduct MakeInnerProduct(const dnnl::engine& eng, int batch, int m, int n, int k, 
                        dnnl::memory::data_type src_dt, dnnl::memory::data_type weights_dt,
                        dnnl::memory::data_type bias_dt, dnnl::memory::data_type dst_dt,
                        const dnnl::primitive_attr& attr = {}) {

    const InnerProductDims dims{batch, m, n, k};

    // plain memory format can be defined using empty strides argument
    dnnl::memory::dims plain{};
    using fmt = dnnl::memory::format_tag;

    const auto src_md     = dnnl::memory::desc(dims.src_tz , src_dt, plain);
    const auto weights_md = dnnl::memory::desc(dims.weights_tz, weights_dt, fmt::any);
    const auto bias_md    = dnnl::memory::desc(dims.bias_tz, bias_dt, plain);
    const auto dst_md     = dnnl::memory::desc(dims.dst_tz, dst_dt, plain);

    return InnerProduct{eng, src_md, weights_md, bias_md, dst_md, attr};
}

template <typename T_src, typename T_wei, typename T_bias, typename T_dst>
InnerProduct MakeInnerProduct(const dnnl::engine& eng, int batch, int m, int n, int k, const dnnl::primitive_attr& attr = {}) {
    const auto src_dt = DnnlDataType<T_src>::value;
    const auto weights_dt = DnnlDataType<T_wei>::value;
    const auto bias_dt = DnnlDataType<T_bias>::value;
    const auto dst_dt = DnnlDataType<T_dst>::value;
    return MakeInnerProduct(eng, batch, m, n, k, src_dt, weights_dt, bias_dt, dst_dt, attr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Softmax
class SoftMax {
public:
    SoftMax(const dnnl::engine& eng,
        const dnnl::memory::desc& data_md,
        int axis,
        const dnnl::primitive_attr& attr = {})
        : prim_{
            dnnl::softmax_forward::primitive_desc{
                dnnl::softmax_forward::desc{dnnl::prop_kind::forward_inference, data_md, axis},
                attr,
                eng}} {}

    SoftMax(const dnnl::primitive& prim) : prim_(prim) {}

    void Compute(dnnl::stream& stm, DataSource& src, dnnl::memory& dst_memory) {
        const auto prim_desc = PrimDesc();
        assert(prim_desc.dst_desc() == dst_memory.get_desc());

        auto src_memory = src.GetData(stm, prim_desc.src_desc());
        prim_.execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_DST, dst_memory } });
        // FIXME(rfsaliev) have to wait due to lifetime of x_memory variables
        stm.wait();
    }

    dnnl::softmax_forward::primitive_desc PrimDesc() const {
        auto c_desc = prim_.get_primitive_desc();
        return dnnl::softmax_forward::primitive_desc{const_cast<dnnl_primitive_desc_t>(c_desc)};
    }

    const dnnl::primitive& Prim() const {
        return prim_;
    }

private:
    dnnl::primitive prim_;
};

inline SoftMax MakeSoftmax(const dnnl::engine& eng, int batch, int m, int n,
                          dnnl::memory::data_type src_dt, int axis,
                          const dnnl::primitive_attr& attr = {}) {
    const dnnl::memory::dims data_tz = {batch, m, n};

    const auto data_md = dnnl::memory::desc(data_tz, src_dt, dnnl::memory::dims{});

    return SoftMax{eng, data_md, axis, attr};
}

template <typename T_data>
SoftMax MakeSoftmax(const dnnl::engine& eng, int batch, int m, int n, int axis, const dnnl::primitive_attr& attr = {}) {
    const auto data_dt = DnnlDataType<T_data>::value;
    return MakeSoftmax(eng, batch, m, n, data_dt, axis, attr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// LayerNorm

class LayerNorm {
public:
    LayerNorm(const dnnl::engine& eng,
        const dnnl::memory::desc& data_md, float epsilon,
        dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift,
        const dnnl::primitive_attr& attr = {})
        : prim_{dnnl::layer_normalization_forward::primitive_desc{
            dnnl::layer_normalization_forward::desc{dnnl::prop_kind::forward_inference, data_md, epsilon, flags}, attr, eng}} {
    }

    LayerNorm(const dnnl::primitive& prim) : prim_(prim) {}

    void Compute(dnnl::stream& stm, DataSource& src, DataSource& scale, DataSource& shift, dnnl::memory& dst_memory) {
        const auto prim_desc = PrimDesc();
        assert(prim_desc.dst_desc() == dst_memory.get_desc());

        const auto src_md = prim_desc.src_desc();
        auto src_memory = src.GetData(stm, src_md);

        const auto scaleshift_md = dnnl::memory::desc{{1, src_md.dims().back()}, src_md.data_type(), dnnl::memory::dims{}};
        auto scale_memory = scale.GetData(stm, scaleshift_md);
        auto shift_memory = shift.GetData(stm, scaleshift_md);

        prim_.execute(stm, {
            { DNNL_ARG_SRC, src_memory },
            { DNNL_ARG_SCALE, scale_memory },
            { DNNL_ARG_SHIFT, shift_memory },
            { DNNL_ARG_DST, dst_memory } });
        // FIXME(rfsaliev) have to wait due to lifetime of x_memory variables
        stm.wait();
    }

    dnnl::layer_normalization_forward::primitive_desc PrimDesc() const {
        auto c_desc = prim_.get_primitive_desc();
        return dnnl::layer_normalization_forward::primitive_desc{const_cast<dnnl_primitive_desc_t>(c_desc)};
    }

    const dnnl::primitive& Prim() const {
        return prim_;
    }

private:
    dnnl::primitive prim_;
};

inline LayerNorm MakeLayerNorm(const dnnl::engine& eng, int batch, int m, int n,
                        dnnl::memory::data_type data_dt, float epsilon,
                        dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift,
                        const dnnl::primitive_attr& attr = {}) {
    const dnnl::memory::dims data_tz = {batch, m, n};

    const auto data_md     = dnnl::memory::desc(data_tz, data_dt, dnnl::memory::dims{});

    return LayerNorm{eng, data_md, epsilon, flags, attr};
}

template <typename T_data>
LayerNorm MakeLayerNorm(const dnnl::engine& eng, int batch, int m, int n, float epsilon,
                 dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift,
                 const dnnl::primitive_attr& attr = {}) {
    const auto data_dt = DnnlDataType<T_data>::value;
    return MakeLayerNorm(eng, batch, m, n, data_dt, epsilon, flags, attr);
}

} // namespace dnnl_wrappers

#endif //__DNNL_OPS__
