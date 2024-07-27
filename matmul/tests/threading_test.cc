#include "test.h"
#include "threading.h"
#include "types.h"

#include <fmt/format.h>

constexpr u32 thread_pool_sizes[] = {
    0, 1, 5, 13, 16, 32, 64
};

consteval bool test_thread_pool_size()
{
    for (const auto &e: thread_pool_sizes)
        if (e > CONFIG_MAX_THREADS)
            return false;

    return true;
}
constexpr bool test_thread_pool_size_result = test_thread_pool_size();
static_assert(test_thread_pool_size_result);

static void work_func(u32 thread_id, std::vector<u32> &outdata, u32 arg)
{
    outdata[thread_id] = thread_id + arg;
}

void test_threading_(bool explicit_exit)
{
    using std::placeholders::_1;

    std::vector<u32> outdata;

    for (const auto thread_pool_size: thread_pool_sizes) {

        /* Prepare output buffer for threads. */
        outdata.resize(thread_pool_size);
        for (auto &e: outdata)
            e = 0xffffffff;

        thread_pool tp;

        tp.resize(thread_pool_size);
        TEST_ASSERT(tp.num_threads() == thread_pool_size);
        TEST_ASSERT(tp.wctx.submit_mask.count() == thread_pool_size);

        const auto offset = 1337;
        tp.schedule(std::bind(work_func, _1, std::ref(outdata), offset));
        tp.sync();

        u32 thread_id = 0;
        for (const auto &e: outdata) {
            TEST_ASSERT(e == thread_id + offset);
            ++thread_id;
        }

        if (explicit_exit)
            tp.exit_threads();
    }
}

void test_threading(bool explicit_exit)
{
    for (u32 i = 0; i < 128; ++i)
        test_threading_(explicit_exit);
}

