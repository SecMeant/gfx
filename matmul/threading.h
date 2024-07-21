#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <bitset>

#include "types.h"

constexpr bool CONFIG_THREAD_TRACE = false;
constexpr u32 CONFIG_MAX_THREADS = 64;

class thread_pool {
private:
    friend void test_threading_(bool);

private:
    using ThreadType = std::thread;
    using WorkType = std::function<void(u32)>;
    using ContainerType = std::vector<ThreadType>;
    using ExecStateBits = std::bitset<CONFIG_MAX_THREADS>;

    struct work_context {
        std::mutex m;
        std::condition_variable cv_submitted;
	std::condition_variable cv_finished;

        /* TODO: bitsets are probably not a good idea for cache false sharing. */
        /* Set for each thread_id that haven't finished last submitted work. */
        ExecStateBits bits_pending;

        /* Mask used to set the bits for all the threads when submitting work. */
        ExecStateBits submit_mask;

        /* Null if exit requested and bits_pending for a thread is set. */
        std::function<void(u32)> work;
    };

    static void idle(u32 thread_id, work_context *wctx);

    void exit_threads();

public:
    thread_pool() = default;
    ~thread_pool() { this->exit_threads(); }

    void schedule(WorkType work);
    void sync();
    void resize(u32 num_threads);
    u32  num_threads() { return this->threads.size(); }

private:
    ContainerType threads;
    work_context wctx;
};
