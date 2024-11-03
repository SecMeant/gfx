#pragma once

#include "mat.h"

#include <stdio.h>

inline void print_mat(matview_i64_t mv)
{
    for (u32 y = 0; y < mv.height; ++y) {

        putchar('{');
        for (u32 x = 0; x < mv.width; ++x) {
            printf("%016lX,", mv[x,y]);
        }
        putchar('}');
        putchar(',');
        putchar('\n');

    }
}
