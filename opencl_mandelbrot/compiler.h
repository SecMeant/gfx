#pragma once

#include <type_traits>

static inline void __attribute__((always_inline))
barrier()
{
        asm("" ::: "memory");
}

template <typename T>
constexpr auto
underlying_cast(T&& e)
{
        return static_cast<std::underlying_type_t<std::remove_reference_t<T>>>(e);
}

