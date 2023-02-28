// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef BERT_PROFILER_HPP_
#define BERT_PROFILER_HPP_

#include <cassert>
#include <functional>
#include <map>
#include <string>
#include <sys/time.h>

class BertProfiler {
    // Re-use oneDNN time measurement algorithm
    static double get_msec() {
        struct timeval time;
        gettimeofday(&time, nullptr);
        return 1e+3 * time.tv_sec + 1e-3 * time.tv_usec;
    }

public:
    template <typename T = double>
    struct Counter
    {
        Counter()
        : iterations{0}
        , min{0}
        , max{0}
        , total{0}
        {};

        void Reset() {
            iterations = 0;
            min = 0;
            max = 0;
            total = 0;
        }

        void Lap(T value) noexcept {
            assert(value >= 0);
            // do not init min=std::numeric_limits<T>::max() for user-friendly reports
            min = (min == 0 || value < min) ? value : min;
            max = value > max ? value : max;
            total += value;
            ++iterations;
        }

        size_t iterations;
        T min;
        T max;
        T total;
    };

    BertProfiler() : enabled_{false} {}

    // Start/Reset profiling optional names to pre-allocate counters map
    void Start(const std::vector<std::string>& names = {}) {
        counters_.clear();
        // 'std::transform()' usage code is more complicated than below:
        for (auto& name : names) {
            counters_.emplace(name, Counter<double>{});
        }
        enabled_ = true;
    }

    void Stop() {
        enabled_ = false;
    }

    void Resume() {
        enabled_ = true;
    }

    void Profile(const std::string& name, const std::function<void()>& f) {
        if (!enabled_) {
            f();
            return;
        }

        auto start_ms = get_msec();
        f();
        double duration_ms = get_msec() - start_ms;
        counters_[name].Lap(duration_ms);
    }

    bool enabled_;
    // (rfsaliev) for small data std::map is pretty effective
    // std::vector would be more effective but more code to write :)
    std::map<std::string, Counter<double>> counters_;
};

#endif // BERT_PROFILER_HPP_
