// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __DNNL_COMMON__
#define __DNNL_COMMON__

#include "dnnl.hpp"

#include <sstream>
#include <string>
#include <unordered_map>
#include <memory>
#include <sstream>

using bfloat16 = std::uint16_t;

template <class T> struct DnnlDataType;

template <> struct DnnlDataType<float> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::f32;
};

template <> struct DnnlDataType<bfloat16> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::bf16;
};

template <> struct DnnlDataType<int8_t> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::s8;
};

template <> struct DnnlDataType<uint8_t> {
    static constexpr dnnl::memory::data_type value = dnnl::memory::data_type::u8;
};

// TODO(rfsaliev) Replace MemoryAccessor with dnnl::reorder functionality to read-write dnnl::memory
template <class T>
class MemoryAccessor {
public:
    MemoryAccessor(dnnl::memory mem)
        : mem_{std::move(mem)}
        , ptr_{mem_ ? mem_.map_data<T>() : nullptr}{}

    ~MemoryAccessor() {
        if (ptr_ && mem_) {
            mem_.unmap_data(ptr_);
        }
    }

    T* Data() { return ptr_; }
private:
    dnnl::memory mem_;
    T* ptr_ = nullptr;

    // Non copyable
    MemoryAccessor(const MemoryAccessor&) = delete;
    MemoryAccessor& operator=(const MemoryAccessor&) = delete;
};

class DnnlCommon {
    public:
    DnnlCommon() : eng(dnnl::engine::kind::cpu, 0), eng_stream(eng) {}

    dnnl::stream& getEngineStream(){
        return eng_stream;
    };

    dnnl::engine& getEngine(){
        return eng;
    }

    private:
    dnnl::engine eng;
    dnnl::stream eng_stream;
};

#endif
