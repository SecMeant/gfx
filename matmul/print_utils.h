#pragma once

#include "matview.h"

#include <stdio.h>

void print_mat(matview_t mv)
{
    for (u32 y = 0; y < mv.height; ++y) {
        for (u32 x = 0; x < mv.width; ++x) {
            printf("%016X ", mv.get(x, y));
        }

        putchar('\n');
    }
}
