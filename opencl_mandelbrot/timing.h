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
        [underlying_cast(clock_state_t::idle)] = "idle",
        [underlying_cast(clock_state_t::started)] = "started",
        [underlying_cast(clock_state_t::finished)] = "finished",
    });

    constexpr static const char*
    state_to_str(clock_state_t state)
    {
        return state_to_str_array.at(underlying_cast(state));
    }

public:
    timeit_t(const char *name)
    : state(clock_state_t::idle)
    {
        strncpy(std::data(this->name), name, std::size(this->name) - 1);
        this->start();
    }

    timeit_t(const timeit_t &other) = default;

    timeit_t(timeit_t &&other) = default;

    timeit_t& operator=(const timeit_t &other) = default;

    timeit_t& operator=(timeit_t &&other) = default;

    ~timeit_t()
    {
        if (this->state != clock_state_t::finished)
            panic("timeit: destroying non-finished clock (%s, %s)\n",
                  this->name, state_to_str(this->state));
    }

    void
    start()
    {
        if (this->state != clock_state_t::idle) [[unlikely]]
            panic("timeit: tried to start a non-idle clock\n");

        barrier();

        this->time_start = ClockType::now();
        this->state = clock_state_t::started;
    }

    void
    stop()
    {
        if (this->state != clock_state_t::started) [[unlikely]]
            panic("timeit: tried to stop a non-started clock\n");

        barrier();
        // TODO: should we also do HW barrier?

        this->time_end = ClockType::now();
        this->state = clock_state_t::finished;
    }

    const char*
    get_name() const
    {
        return this->name;
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
    char name[16];
    clock_state_t state;
    TimePoint time_start;
    TimePoint time_end;
};

/* 
 * We need some kind of place for runners to store their timing information for
 * each stage they care about. Each runner can have different stages and so we
 * just collect a collection of such structures and later print them to the
 * user.
 *
 * Maybe it should be defined elsewhere.
 */
using timing_info_t = std::vector<timeit_t>;

