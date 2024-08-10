#pragma once

#include "config.h"
#include "compiler.h"
#include "panic.h"
#include "types.h"

#include <array>
#include <chrono>

class timeit_t {
public:
    using ClockType = std::chrono::high_resolution_clock;
    using TimePoint = ClockType::time_point;
    using Duration = ClockType::duration;

private:
    enum class clock_state_t : u8 {
        idle,
        started,
        finished,
    };

private:
    constexpr static std::array state_to_str_array = std::to_array({
        [ucast(clock_state_t::idle)] = "idle",
        [ucast(clock_state_t::started)] = "started",
        [ucast(clock_state_t::finished)] = "finished",
    });

    constexpr static const char*
    state_to_str(clock_state_t state)
    {
        return state_to_str_array.at(ucast(state));
    }

public:
    timeit_t(): state(clock_state_t::idle) {}

    timeit_t(const timeit_t &other) = default;

    timeit_t(timeit_t &&other) = default;

    timeit_t& operator=(const timeit_t &other) = default;

    timeit_t& operator=(timeit_t &&other) = default;

    ~timeit_t() = default;

    void
    start()
    {
        barrier();

        this->time_start = ClockType::now();
        this->state = clock_state_t::started;
    }

    void
    stop()
    {
        barrier();
        // TODO: should we also do HW barrier?

        this->time_end = ClockType::now();
        this->state = clock_state_t::finished;
    }

    Duration
    get_duration() const
    {
        return this->time_end - this->time_start;
    }

    u64
    get_duration_micro() const
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(this->get_duration()).count();
    }

    u64
    get_duration_nano() const
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(this->get_duration()).count();
    }

private:
    clock_state_t state;
    TimePoint time_start;
    TimePoint time_end;
};

