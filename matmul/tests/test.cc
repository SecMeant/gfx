#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <unistd.h>

#include <fmt/format.h>

#include "test.h"
#include "mat.h"
#include "print_utils.h"
#include "get_type_name.h"
#include "threading.h"

constexpr bool VERBOSE = true;

static void print_mat_mul_context_(
    const u32 x,
    const u32 y,
    const matview_t actual,
    const matview_t expected,
    const matview_t lhs,
    const matview_t rhs
) {
    assert(lhs.height == rhs.width);
    assert(lhs.height != 0);

    u32 idx = 0;

    for(;;) {
        fmt::print("{} * {} ", lhs[idx, y], rhs[x, idx]);

        ++idx;

        if (idx == lhs.height)
            break;

        fmt::print("+ ");
    }

    fmt::print(" = {} != {}\n", actual[x, y], expected[x, y]);
}

/* Returns number of mismatching elements capped by max_num_miscmp. */
static int mat_compare_(
    const matview_t actual,
    const matview_t expected,
    const matview_t lhs,
    const matview_t rhs,
    const mat_op op
) {
    constexpr u32 max_num_miscmp = 4;
    u32 cur_num_miscmp = 0;

    /* For now we only support square matrices */
    if (!mat_dim_match(actual, expected))
        throw test_failure("Matrix dimensions do not match");

    const auto height = actual.height;
    const auto width = actual.width;

    for (u32 y = 0; y < height; ++y) {

        for (u32 x = 0; x < width; ++x) {

            const auto va = actual[x, y];
            const auto ve = expected[x, y];

            if (va != ve) {
                fmt::print("miscompare at ({}, {}): {} != {}\n", x, y, va, ve);

                if constexpr (VERBOSE)
                    if (op == mat_op::mul)
                        print_mat_mul_context_(x, y, actual, expected, lhs, rhs);

                ++cur_num_miscmp;
                if (cur_num_miscmp >= max_num_miscmp)
                    return cur_num_miscmp;
            }

        }

    }

    return cur_num_miscmp;
}

void mat_compare_or_fail(
    const char *test_name,
    const matview_t actual,
    const matview_t expected,
    const matview_t lhs,
    const matview_t rhs,
    const mat_op op
) {
    const int failed = mat_compare_(actual, expected, lhs, rhs, op);

    if (failed > 0)
        throw test_failure(fmt::format("{}: data miscompare", test_name));
}

static void test_matrix_simple_add()
{
    using init_t = mat_t::InitializerType;

    const init_t lhs_data = {
        {1,  2,  3,  4 },
        {11, 12, 13, 14},
        {21, 22, 23, 24},
    };

    const init_t rhs_data = {
        {4,  2, 3,  5},
        {87, 4, 16, 4},
        {12, 2, 4,  4},
    };

    const init_t expected_data = {
        {5,  4,  6,  9 },
        {98, 16, 29, 18},
        {33, 24, 27, 28},
    };

    const auto lhs = mat_t(lhs_data);
    const auto rhs = mat_t(rhs_data);

    const auto out = mat_add_cpu(lhs, rhs);

    for (size_t y = 0; y < lhs_data.size(); ++y) {
        for (size_t x = 0; x < lhs_data[0].size(); ++x) {
            TEST_ASSERT((out[x,y] == expected_data[y][x]));
            TEST_ASSERT((lhs[x,y] + rhs[x,y] == lhs_data[y][x] + rhs_data[y][x]));
        }
    }
}

static void test_matrix_simple_mul()
{
    using init_t = mat_t::InitializerType;

    const init_t lhs_data = {
        {1,  2,  3,  4 },
        {11, 12, 13, 14},
        {21, 22, 23, 24},
        {45, 98, 66, 0 },
    };

    const init_t rhs_data = {
        {4,  2, 3,  5},
        {87, 4, 16, 4},
        {12, 2, 4,  4},
        {4,  3, 1,  9},
    };

    const init_t expected_data = {
        {230,  28,  51,   61 },
        {1300, 138, 291,  281},
        {2370, 248, 531,  501},
        {9498, 614, 1967, 881}
    };

    const auto lhs = mat_t(lhs_data);
    const auto rhs = mat_t(rhs_data);

    const auto out = mat_mul_cpu(lhs, rhs);

    for (size_t y = 0; y < lhs_data.size(); ++y)
        for (size_t x = 0; x < lhs_data[0].size(); ++x)
            TEST_ASSERT((out[x,y] == expected_data[y][x]));
}

static void test_matrix_simple_strassen_mul()
{
    using init_t = mat_t::InitializerType;

    const init_t lhs_data = {
        {1,  2,  3,  4 , 1,  2,  3,  4 },
        {11, 12, 13, 14, 11, 12, 13, 14},
        {21, 22, 23, 24, 21, 22, 23, 24},
        {45, 98, 66, 0 , 45, 98, 66, 0 },
        {1,  2,  3,  4 , 1,  2,  3,  4 },
        {11, 12, 13, 14, 11, 12, 13, 14},
        {21, 22, 23, 24, 21, 22, 23, 24},
        {45, 98, 66, 0 , 45, 98, 66, 0 },
    };

    const init_t rhs_data = {
        {4,  2, 3,  5, 4,  2, 3,  5},
        {87, 4, 16, 4, 87, 4, 16, 4},
        {12, 2, 4,  4, 12, 2, 4,  4},
        {4,  3, 1,  9, 4,  3, 1,  9},
        {4,  2, 3,  5, 4,  2, 3,  5},
        {87, 4, 16, 4, 87, 4, 16, 4},
        {12, 2, 4,  4, 12, 2, 4,  4},
        {4,  3, 1,  9, 4,  3, 1,  9},
    };

    const auto lhs = mat_t(lhs_data);
    const auto rhs = mat_t(rhs_data);

    const auto out0 = strassen_cpu(lhs, rhs);
    const auto out1 = mat_mul_cpu(lhs, rhs);

    for (u32 y = 0; y < out0.height; ++y)
        for (u32 x = 0; x < out0.width; ++x)
            TEST_ASSERT((out0[x, y] == out1[x, y]));
}

void test_matrix_vs_pytorch(const char *safetensors_path);
void test_threading(bool explicit_exit);

static std::queue<std::string> test_status;
static std::mutex test_status_mtx;
static std::condition_variable test_status_cv;

static void RUN_TEST(const test& test)
{
    std::string status;

    try {
        test_stats.num_tests.fetch_add(1, std::memory_order_relaxed);
        test.func();
        status = fmt::format("{}: " CLR_GREEN "OK\n" CLR_RESET, test.name);
    } catch (const std::exception &e) {
        status = fmt::format("{}: " CLR_RED "Failed: {}\n" CLR_RESET, test.name, e.what());
        test_stats.num_failed.fetch_add(1, std::memory_order_relaxed);
    }

    /* lock guard */ {
        std::unique_lock lck(test_status_mtx);
        test_status.push(std::move(status));
        test_status_cv.notify_all();
    }
}

int main()
{
    const std::vector<test> tests {
        test{
            .name = "test_threading(explicit_exit = 0)",
            .func = std::bind(test_threading, false),
        },
        {
            .name = "test_threading(explicit_exit = 1)",
            .func = std::bind(test_threading, true),
        },
        {
            .name = "test_matrix_simple_add",
            .func = std::bind(test_matrix_simple_add),
        },
        {
            .name = "test_matrix_simple_mul",
            .func = std::bind(test_matrix_simple_mul),
        },
        {
            .name = "test_matrix_simple_strassen_mul",
            .func = std::bind(test_matrix_simple_strassen_mul),
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_4x4.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch, CONFIG_TEST_FILES_PATH "pytorch_4x4.safetensors"),
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_64x64.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch, CONFIG_TEST_FILES_PATH "pytorch_64x64.safetensors"),
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_128x128.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch, CONFIG_TEST_FILES_PATH "pytorch_128x128.safetensors"),
        },
    };

    long int num_cpus_online = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cpus_online <= 0)
        num_cpus_online = 4;

    const u32 num_threads = std::min(tests.size(), static_cast<size_t>(num_cpus_online));
    thread_pool threads(num_threads);

    threads.schedule([&](u32 thread_id) {
        const u32 batch_size = num_threads;
        u32 job_id = thread_id;

        while (job_id < tests.size()) {
            RUN_TEST(tests[job_id]);
            job_id += batch_size;
        }
    });

    const u32 job_count = tests.size();
    for (u32 processed = 0; processed < job_count;) {
        std::unique_lock lck(test_status_mtx);
        test_status_cv.wait(lck);

        while (test_status.size()) {
            const auto &status = test_status.front();
            printf(status.c_str());
            test_status.pop();
            fflush(stdout);
            ++processed;
        }
    }

    threads.sync();

    const auto tests_run = test_stats.num_tests.load(std::memory_order_relaxed);
    const auto tests_failed = test_stats.num_failed.load(std::memory_order_relaxed);
    printf("\nTests run   : %zu\nTests failed: %zu\n", tests_run, tests_failed);

    return 0;
}
