#include "interrupt.h"

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>

static void interrupt_handler(int s)
{
    if (s == SIGINT)
        interrupt_requested.store(true, std::memory_order_relaxed);
}

int register_interrupt_handler()
{
    struct sigaction sa;

    sa.sa_handler = interrupt_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    if (sigaction(SIGINT, &sa, NULL) == -1) {
        return errno;
    }

    return 0;
}

