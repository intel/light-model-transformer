// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_DATA__
#define __DNNL_DATA__

#include "dnnl_common.h"
#include "dnnl_attr.hpp"

#include <string>

namespace dnnl_wrappers {

class DataSource {
public:
    DataSource(const dnnl::memory& mem = {})
        : mem_{mem}
        , attr_{nullptr} {}

    DataSource(const dnnl::memory& mem, const dnnl::primitive_attr& attr)
        : mem_{mem}
        , attr_{new dnnl::primitive_attr(attr)} {}

    DataSource(const DataSource& other) = default;
    DataSource(DataSource&& other) = default;
    DataSource& operator=(const DataSource& other) = default;
    DataSource& operator=(DataSource&& other) = default;
    virtual ~DataSource() = default;

    virtual dnnl::memory GetData(dnnl::stream& stm, const dnnl::memory::desc& md) {
        if (!mem_) {
             return mem_;
        }

        if (!attr_ && mem_.get_engine() == stm.get_engine() && mem_.get_desc() == md) {
            return mem_;
        }

        dnnl::memory result{md, stm.get_engine()};

        dnnl::reorder rdr{mem_, result, attr_ ? *attr_ : dnnl::primitive_attr{}};
        rdr.execute(stm, mem_, result);
        return result;
    }

private:
    dnnl::memory mem_;
    // use pointer to make it optional
    std::unique_ptr<dnnl::primitive_attr> attr_;
};

class CachedDataSource : public DataSource {
public:
    using DataSource::DataSource;

    dnnl::memory GetData(dnnl::stream& stm, const dnnl::memory::desc& md) override {
        if (!cached_mem_ || cached_mem_.get_engine() != stm.get_engine() || cached_mem_.get_desc() != md) {
            cached_mem_ = DataSource::GetData(stm, md);
        }
        return cached_mem_;
    }

private:
    dnnl::memory cached_mem_;
};

inline dnnl::memory::format_tag PlainFormatTag(size_t ndims, bool trans = false) {
    using ft = dnnl::memory::format_tag;
    switch (ndims) {
        case 1: return ft::a;
        case 2: return trans ? ft::ba : ft::ab;
        case 3: return trans ? ft::acb : ft::abc;
        case 4: return trans ? ft::abdc : ft::abcd;
        default: return ft::undef;
    }
}

template <class T>
dnnl::memory AttachMemory(const dnnl::engine& eng, dnnl::memory::dims dims, T* data, bool trans = false) {
    const auto dt = DnnlDataType<T>::value;
    dnnl::memory::desc md{dims, dt, PlainFormatTag(dims.size(), trans)};
    return dnnl::memory{md, eng, data};
}

template <class T>
dnnl::memory CloneMemory(const dnnl::engine& eng, dnnl::stream& stm, dnnl::memory::dims dims, const T* data, bool trans = false) {
    const auto dt = DnnlDataType<T>::value;
    auto src = AttachMemory(eng, dims, const_cast<T*>(data), trans);
    dnnl::memory::desc md{dims, dt, dnnl::memory::dims{}};
    dnnl::memory dst{md, stm.get_engine()};
    dnnl::reorder{src, dst}.execute(stm, src, dst);
    stm.wait();
    return dst;
}

DataSource ScaledData(const dnnl::memory& mem, float scale) {
    return scale != BuildAttrs::noScale
        ? DataSource(mem, BuildAttrs().Scale(scale))
        : DataSource(mem);
}

CachedDataSource ScaledCachedData(const dnnl::memory& mem, float scale) {
    return  scale != BuildAttrs::noScale
        ? CachedDataSource(mem, BuildAttrs().Scale(scale))
        : CachedDataSource(mem);
}

class GCachedDataSource : public DataSource {
public:
    GCachedDataSource(const std::string& key, map_mem_t& g_memory, const dnnl::memory& mem)
        : DataSource{mem}, key_{key}, g_memory_{g_memory} {}

    GCachedDataSource(const std::string& key, map_mem_t& g_memory, const dnnl::memory& mem, const dnnl::primitive_attr& attr)
        : DataSource{mem, attr}, key_{key}, g_memory_{g_memory} {}

    dnnl::memory GetData(dnnl::stream& stm, const dnnl::memory::desc& md) override {
        auto it_memory_created = g_memory_.find(key_);
        if (it_memory_created != g_memory_.end()) {
            return it_memory_created->second;
        }

        auto result = DataSource::GetData(stm, md);
        g_memory_.emplace(key_, result);
        return result;
    }

private:
    std::string key_;
    map_mem_t& g_memory_;
};

GCachedDataSource ScaledCachedData(const std::string& key, map_mem_t& g_memory, const dnnl::memory& mem, float scale) {
    return scale != BuildAttrs::noScale
        ? GCachedDataSource(key, g_memory, mem, BuildAttrs().Scale(scale))
        : GCachedDataSource(key, g_memory, mem);
}

} // namespace dnnl_wrappers

#endif //__DNNL_DATA__
