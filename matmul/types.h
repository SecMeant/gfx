#pragma once
#include <stdint.h>
#include <utility>

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;

template <typename T, typename U>
auto rcast(U&& arg)
{
    return reinterpret_cast<T>(std::forward<U>(arg));
}

template <typename T, typename U>
auto scast(U&& arg)
{
    return static_cast<T>(std::forward<U>(arg));
}

