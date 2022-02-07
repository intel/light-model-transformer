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
    MatMulDims(int m, int n, int k, bool bias_2d = false)
        : src_tz{m, k}
        , weights_tz{k, n}
        , bias_tz{bias_2d ? m : 1, n}
        , dst_tz{m, n} {}

    dnnl::memory::dims src_tz;
    dnnl::memory::dims weights_tz;
    dnnl::memory::dims bias_tz;
    dnnl::memory::dims dst_tz;
};

inline MatMul CachedMatMul(const std::string& key, DnnlCommon& dnnl_context,
                           const dnnl::memory::desc& src_md,
                           const dnnl::memory::desc& weights_md,
                           const dnnl::memory::desc& bias_md,
                           const dnnl::memory::desc& dst_md,
                        const dnnl::primitive_attr& attr = {}) {
    auto& g_prim = dnnl_context.get_g_prim();
    auto it_prim_created = g_prim.find(key);
    if (it_prim_created == g_prim.end())
    {
        MatMul result{dnnl_context.getEngine(), src_md, weights_md, bias_md, dst_md, attr};
        g_prim.emplace(key, result.Prim());
        return result;
    }
    return MatMul(it_prim_created->second);
}

inline MatMul MakeMatMul(const dnnl::engine& eng, int m, int n, int k, 
                        dnnl::memory::data_type src_dt, dnnl::memory::data_type weights_dt,
                        dnnl::memory::data_type bias_dt, dnnl::memory::data_type dst_dt,
                        const dnnl::primitive_attr& attr = {}) {

    const MatMulDims dims{m, n, k};

    const auto src_fmt = dnnl::memory::format_tag::ab;
    const auto bias_fmt = dnnl::memory::format_tag::ab;
    const auto dst_fmt =  dnnl::memory::format_tag::ab;
    const auto weights_fmt = dnnl::memory::format_tag::any;

    const auto src_md     = dnnl::memory::desc(dims.src_tz , src_dt, src_fmt);
    const auto weights_md = dnnl::memory::desc(dims.weights_tz, weights_dt, weights_fmt);
    const auto bias_md    = dnnl::memory::desc(dims.bias_tz, bias_dt, bias_fmt);
    const auto dst_md     = dnnl::memory::desc(dims.dst_tz, dst_dt, dst_fmt);

    return MatMul{eng, src_md, weights_md, bias_md, dst_md, attr};
}

inline MatMul CachedMatMul(const std::string& key, DnnlCommon& dnnl_context, int m, int n, int k, 
                        dnnl::memory::data_type src_dt, dnnl::memory::data_type weights_dt,
                        dnnl::memory::data_type bias_dt, dnnl::memory::data_type dst_dt,
                        const dnnl::primitive_attr& attr = {}) {
    auto& g_prim = dnnl_context.get_g_prim();
    auto it_prim_created = g_prim.find(key);
    if (it_prim_created == g_prim.end())
    {
        auto result = MakeMatMul(dnnl_context.getEngine(), m, n, k, src_dt, weights_dt, bias_dt, dst_dt, attr);
        g_prim.emplace(key, result.Prim());
        return result;
    }
    return MatMul(it_prim_created->second);
}

template <typename T_src, typename T_wei, typename T_bias, typename T_dst>
MatMul MakeMatMul(const dnnl::engine& eng, int m, int n, int k, const dnnl::primitive_attr& attr = {}) {
    const auto src_dt = DnnlDataType<T_src>::value;
    const auto weights_dt = DnnlDataType<T_wei>::value;
    const auto bias_dt = DnnlDataType<T_bias>::value;
    const auto dst_dt = DnnlDataType<T_dst>::value;
    return MakeMatMul(eng, m, n, k, src_dt, weights_dt, bias_dt, dst_dt, attr);
}

template <typename T_src, typename T_wei, typename T_bias, typename T_dst>
MatMul CachedMatMul(const std::string& key, DnnlCommon& dnnl_context, int m, int n, int k, const dnnl::primitive_attr& attr = {}) {
    const auto src_dt = DnnlDataType<T_src>::value;
    const auto weights_dt = DnnlDataType<T_wei>::value;
    const auto bias_dt = DnnlDataType<T_bias>::value;
    const auto dst_dt = DnnlDataType<T_dst>::value;
    return CachedMatMul(key, dnnl_context, m, n, k, src_dt, weights_dt, bias_dt, dst_dt, attr);
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

inline SoftMax MakeSoftmax(const dnnl::engine& eng, int m, int n,
                          dnnl::memory::data_type src_dt, int axis,
                          const dnnl::primitive_attr& attr = {}) {
    const dnnl::memory::dims data_tz = { m, n };

    const auto data_format = dnnl::memory::format_tag::ab;

    const auto data_md = dnnl::memory::desc({ data_tz }, src_dt, data_format);

    return SoftMax{eng, data_md, axis, attr};
}

inline SoftMax CachedSoftmax(const std::string& key, DnnlCommon& dnnl_context, int m, int n,
                            dnnl::memory::data_type src_dt, int axis,
                            const dnnl::primitive_attr& attr = {}) {
    auto& g_prim = dnnl_context.get_g_prim();
    auto it_prim_created = g_prim.find(key);
    if (it_prim_created == g_prim.end())
    {
        auto result = MakeSoftmax(dnnl_context.getEngine(), m, n, src_dt, axis, attr);
        g_prim.emplace(key, result.Prim());
        return result;
    }
    return SoftMax{it_prim_created->second};
}

template <typename T_data>
SoftMax MakeSoftmax(const dnnl::engine& eng, int m, int n, int axis, const dnnl::primitive_attr& attr = {}) {
    const auto data_dt = DnnlDataType<T_data>::value;
    return MakeSoftmax(eng, m, n, data_dt, axis, attr);
}

template <typename T_data>
SoftMax CachedSoftmax(const std::string& key, DnnlCommon& dnnl_context, int m, int n, int axis, const dnnl::primitive_attr& attr = {}) {
    const auto data_dt = DnnlDataType<T_data>::value;
    return CachedSoftmax(key, dnnl_context, m, n, data_dt, axis, attr);
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

        const auto scaleshift_md = dnnl::memory::desc{{1, src_md.dims().at(1)}, src_md.data_type(), dnnl::memory::dims{}};
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

inline LayerNorm MakeLayerNorm(const dnnl::engine& eng, int m, int n,
                        dnnl::memory::data_type data_dt, float epsilon,
                        dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift,
                        const dnnl::primitive_attr& attr = {}) {
    const dnnl::memory::dims data_tz = { m, n };

    const auto data_fmt = dnnl::memory::format_tag::ab;

    const auto data_md     = dnnl::memory::desc({ data_tz }, data_dt, data_fmt);

    return LayerNorm{eng, data_md, epsilon, flags, attr};
}

inline LayerNorm CachedLayerNorm(const std::string& key, DnnlCommon& dnnl_context, int m, int n,
                        dnnl::memory::data_type data_dt, float epsilon,
                        dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift,
                        const dnnl::primitive_attr& attr = {}) {
    auto& g_prim = dnnl_context.get_g_prim();
    auto it_prim_created = g_prim.find(key);
    if (it_prim_created == g_prim.end())
    {
        auto result = MakeLayerNorm(dnnl_context.getEngine(), m, n, data_dt, epsilon, flags, attr);
        g_prim.emplace(key, result.Prim());
        return result;
    }
    return LayerNorm(it_prim_created->second);
}

template <typename T_data>
LayerNorm MakeLayerNorm(const dnnl::engine& eng, int m, int n, float epsilon,
                 dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift,
                 const dnnl::primitive_attr& attr = {}) {
    const auto data_dt = DnnlDataType<T_data>::value;
    return MakeLayerNorm(eng, m, n, data_dt, epsilon, flags, attr);
}

template <typename T_data>
LayerNorm CachedLayerNorm(const std::string& key, DnnlCommon& dnnl_context, int m, int n, float epsilon,
                 dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift,
                 const dnnl::primitive_attr& attr = {}) {
    const auto data_dt = DnnlDataType<T_data>::value;
    return CachedLayerNorm(key, dnnl_context, m, n, data_dt, epsilon, flags, attr);
}

} // namespace dnnl_wrappers

#endif //__DNNL_OPS__
