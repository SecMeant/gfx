#pragma once

#include <atomic>

inline std::atomic_bool interrupt_requested = ATOMIC_FLAG_INIT;

inline bool should_exit()
{
    return interrupt_requested.load(std::memory_order_relaxed);
}

int register_interrupt_handler();
