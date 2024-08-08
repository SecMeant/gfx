#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

inline void panic(const char *format, ...)
{
    va_list args;
    va_start(args, format);

    vfprintf(stderr, format, args);

    exit(1);
}
