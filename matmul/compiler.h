#pragma once

static inline void __attribute__((always_inline))
barrier()
{
        asm("" ::: "memory");
}

