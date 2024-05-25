#pragma once

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

inline void panic(const char *format, ...)
{
    va_list args;
    va_start(args, format);

    vprintf(format, args);

    exit(1);
}
