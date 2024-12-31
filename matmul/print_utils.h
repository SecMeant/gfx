#pragma once

#include "mat.h"

#include <stdio.h>

template <typename MatrixType>
struct print_mat_format_;

template<> struct print_mat_format_<matview_i64_t> { constexpr static decltype(auto) format = "%016lX,"; };
template<> struct print_mat_format_<matview_f32_t> { constexpr static decltype(auto) format = "% 12.2f,"; };

template <typename T>
constexpr static decltype(auto) print_mat_format = print_mat_format_<T>::format;

template <typename ParentType>
inline void print_mat(matview_base_t<ParentType> mv)
{
    for (u32 y = 0; y < mv.height; ++y) {

        putchar('{');
        for (u32 x = 0; x < mv.width; ++x) {
            printf(print_mat_format<decltype(mv)>, mv.at(x,y));
        }
        putchar('}');
        putchar(',');
        putchar('\n');

    }
}

template <typename ValueType>
inline void print_mat(const mat_base_t<ValueType> &m)
{
    print_mat(matview_base_t(m));
}
