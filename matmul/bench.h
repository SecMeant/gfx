#pragma once

#include <vector>
#include <tuple>
#include <chrono>
#include <mutex>

class benchinfo_t {
public:
    using ClockType = std::chrono::high_resolution_clock;
    using TimePoint = ClockType::time_point;
    using Duration = ClockType::duration;

    struct entry_t {
        std::string name;
        Duration duration;
    };

    using BenchmarkEntries = std::vector<entry_t>;

    benchinfo_t() = default;

    void add(std::string name, Duration duration)
    {
        std::unique_lock lck(this->mtx);
        this->entries.emplace_back(std::move(name), std::move(duration));
    }

    auto consume_entries()
    {
        std::unique_lock lck(this->mtx);

        BenchmarkEntries ret = std::move(this->entries);

        return ret;
    }

private:
    BenchmarkEntries entries;
    std::mutex mtx;
} inline benchinfo;

