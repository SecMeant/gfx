#include <thread>
#include <vector>
#include <queue>
#include <cmath>
#include <condition_variable>
#include <unistd.h>

#include <fmt/format.h>

#include "matmul_cuda.h"

#include "test.h"
#include "mat.h"
#include "print_utils.h"
#include "get_type_name.h"
#include "threading.h"
#include "timing.h"
#include "bench.h"
#include "options.h"
#include "interrupt.h"

constexpr bool VERBOSE = true;

template <typename MatViewType>
void print_mat_mul_context_(
    const u32 x,
    const u32 y,
    const MatViewType actual,
    const MatViewType expected,
    const MatViewType lhs,
    const MatViewType rhs
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
template <typename MatViewType>
int mat_compare_(
    const MatViewType actual,
    const MatViewType expected,
    const MatViewType lhs,
    const MatViewType rhs,
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

template <typename MatViewType>
void mat_compare_or_fail_(
    const char *test_name,
    const MatViewType actual,
    const MatViewType expected,
    const MatViewType lhs,
    const MatViewType rhs,
    const mat_op op
) {
    const int failed = mat_compare_(actual, expected, lhs, rhs, op);

    if (failed > 0)
        throw test_failure(fmt::format("{}: data miscompare", test_name));
}

void mat_compare_or_fail(
    const char *test_name,
    const matview_i64_t actual,
    const matview_i64_t expected,
    const matview_i64_t lhs,
    const matview_i64_t rhs,
    const mat_op op
) {
    mat_compare_or_fail_(test_name, actual, expected, lhs, rhs, op);
}

void mat_compare_or_fail(
    const char *test_name,
    const matview_f32_t actual,
    const matview_f32_t expected,
    const matview_f32_t lhs,
    const matview_f32_t rhs,
    const mat_op op
) {
    mat_compare_or_fail_(test_name, actual, expected, lhs, rhs, op);
}

template <typename MatrixType>
void test_matrix_simple_add()
{
    using init_t = MatrixType::InitializerType;

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

    const auto lhs = MatrixType(lhs_data);
    const auto rhs = MatrixType(rhs_data);

    const auto out = mat_add_cpu(lhs, rhs);

    for (size_t y = 0; y < lhs_data.size(); ++y) {
        for (size_t x = 0; x < lhs_data[0].size(); ++x) {
            TEST_ASSERT((out[x,y] == expected_data[y][x]));
            TEST_ASSERT((lhs[x,y] + rhs[x,y] == lhs_data[y][x] + rhs_data[y][x]));
        }
    }
}

template <typename MatrixType>
void test_matrix_simple_mul()
{
    using init_t = MatrixType::InitializerType;

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

    const auto lhs = MatrixType(lhs_data);
    const auto rhs = MatrixType(rhs_data);

    const auto out = mat_mul_cpu(lhs, rhs);

    for (size_t y = 0; y < lhs_data.size(); ++y)
        for (size_t x = 0; x < lhs_data[0].size(); ++x)
            TEST_ASSERT((out[x,y] == expected_data[y][x]));
}

template <typename MatrixType>
void test_matrix_simple_strassen_mul()
{
    using init_t = MatrixType::InitializerType;

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

    const auto lhs = MatrixType(lhs_data);
    const auto rhs = MatrixType(rhs_data);

    const auto out0 = strassen_cpu(lhs, rhs);
    const auto out1 = mat_mul_cpu(lhs, rhs);

    for (u32 y = 0; y < out0.height; ++y)
        for (u32 x = 0; x < out0.width; ++x)
            TEST_ASSERT((out0[x, y] == out1[x, y]));
}

static void test_matrix_simple_opencl_mul()
{
    using init_t = mat_i64_t::InitializerType;

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

    const auto lhs = mat_i64_t(lhs_data);
    const auto rhs = mat_i64_t(rhs_data);

    const auto out0 = mat_mul_cl(lhs, rhs);
    const auto out1 = mat_mul_cpu(lhs, rhs);

    for (u32 y = 0; y < out0.height; ++y)
        for (u32 x = 0; x < out0.width; ++x)
            TEST_ASSERT((out0[x, y] == out1[x, y]));
}

constexpr static i64 STATUS_LINE_ALIGNMENT = 75;
constexpr static i64 BENCHMARK_LINE_ALIGNMENT = 62;

void test_matrix_vs_pytorch_i32(const char *safetensors_path, test_flags_t flags);
void test_matrix_vs_pytorch_f32(const char *safetensors_path, test_flags_t flags);
void test_threading(bool explicit_exit);

static std::queue<std::string> test_status;
static std::mutex test_status_mtx;
static std::condition_variable test_status_cv;

static void append_time_string_(std::string &out, timeit_t::Duration duration, i64 alignment)
{
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;
    using std::chrono::microseconds;
    using std::chrono::milliseconds;
    using std::chrono::seconds;
    using std::to_string;

    u32 printed = 0;

    const i64 filler = alignment - scast<i64>(out.size());
    if (filler > 0)
        out.append(filler, ' ');

    const auto dur_secs = duration_cast<seconds>(duration);
    if (dur_secs.count() != 0) {
        out += to_string(dur_secs.count());
        out += "s ";
        duration -= dur_secs;
        ++printed;
    }

    const auto dur_ms = duration_cast<milliseconds>(duration);
    if (dur_ms.count() != 0) {
        out += to_string(dur_ms.count());
        out += "ms ";
        duration -= dur_ms;
        ++printed;
    }

    if (printed >= 2)
        return;

    const auto dur_us = duration_cast<microseconds>(duration);
    if (dur_us.count() != 0) {
        out += to_string(dur_us.count());
        out += "us ";
        duration -= dur_us;
        ++printed;
    }

    if (printed >= 2)
        return;

    const auto dur_ns = duration_cast<nanoseconds>(duration);
    if (dur_ns.count() != 0) {
        out += to_string(dur_ns.count());
        out += "ns ";
        duration -= dur_ns;
        ++printed;
    }

    if (printed >= 2)
        return;
}

static void append_time_string(std::string &out, timeit_t::Duration duration, i64 alignment)
{
    append_time_string_(out, duration, alignment);
    out += '\n';
}

static void RUN_TEST(const test& test)
{

    std::string status;

    try {
        test_stats.num_tests.fetch_add(1, std::memory_order_relaxed);

        timeit_t timer;
        timer.start();
        test.func();
        timer.stop();

        status = fmt::format("{}: " CLR_GREEN "OK " CLR_RESET, test.name);
        append_time_string(status, timer.get_duration(), STATUS_LINE_ALIGNMENT);

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

int run_tests()
{
    const std::vector<test> all_tests {
        /* SIMPLE CPU TESTS */
        {
            .name = "test_threading(explicit_exit = 0)",
            .func = std::bind(test_threading, false),
            .group = test_group::i64,
        },
        {
            .name = "test_threading(explicit_exit = 1)",
            .func = std::bind(test_threading, true),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_simple_add_i64",
            .func = std::bind(test_matrix_simple_add<mat_i64_t>),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_simple_mul_i64",
            .func = std::bind(test_matrix_simple_mul<mat_i64_t>),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_simple_strassen_mul_i64",
            .func = std::bind(test_matrix_simple_strassen_mul<mat_i64_t>),
            .group = test_group::i64,
        },

        /* SIMPLE CPU TESTS F32 */
        {
            .name = "test_matrix_simple_add_f32",
            .func = std::bind(test_matrix_simple_add<mat_f32_t>),
            .group = test_group::f32,
        },
        {
            .name = "test_matrix_simple_mul_f32",
            .func = std::bind(test_matrix_simple_mul<mat_f32_t>),
            .group = test_group::f32,
        },
        {
            .name = "test_matrix_simple_strassen_mul_f32",
            .func = std::bind(test_matrix_simple_strassen_mul<mat_f32_t>),
            .group = test_group::f32,
        },


        /* SIMPLE OPENCL TESTS */
        {
            .name = "test_matrix_simple_opencl_mul",
            .func = std::bind(test_matrix_simple_opencl_mul),
            .group = test_group::i64,
        },


        /* SAFETENSORS TESTS */
        {
            .name = "test_matrix_vs_pytorch(pytorch_4x4.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_i32, CONFIG_TEST_FILES_PATH "pytorch_4x4.safetensors",
                              test_flags_t{}),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_64x64.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_i32, CONFIG_TEST_FILES_PATH "pytorch_64x64.safetensors",
                              test_flags_t{}),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_128x128.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_i32, CONFIG_TEST_FILES_PATH "pytorch_128x128.safetensors",
                              test_flags_t{}),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_256x256.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_i32, CONFIG_TEST_FILES_PATH "pytorch_256x256.safetensors",
                              test_flags_t{}),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_512x512.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_i32, CONFIG_TEST_FILES_PATH "pytorch_512x512.safetensors",
                              test_flags_t{
                                  .skip_cpu = 1
                              }),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_1024x1024.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_i32, CONFIG_TEST_FILES_PATH "pytorch_1024x1024.safetensors",
                              test_flags_t{
                                  .skip_cpu = 1
                              }),
            .group = test_group::i64,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_2048x2048.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_i32, CONFIG_TEST_FILES_PATH "pytorch_2048x2048.safetensors",
                              test_flags_t{
                                  .skip_cpu = 1
                              }),
            .group = test_group::i64,
        },


        /* SAFETENSORS TESTS F32 */
        {
            .name = "test_matrix_vs_pytorch(pytorch_4x4_f32.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_f32, CONFIG_TEST_FILES_PATH "pytorch_4x4_f32.safetensors",
                              test_flags_t{}),
            .group = test_group::f32,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_64x64_f32.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_f32, CONFIG_TEST_FILES_PATH "pytorch_64x64_f32.safetensors",
                              test_flags_t{}),
            .group = test_group::f32,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_256x256_f32.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_f32, CONFIG_TEST_FILES_PATH "pytorch_256x256_f32.safetensors",
                              test_flags_t{}),
            .group = test_group::f32,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_512x512_f32.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_f32, CONFIG_TEST_FILES_PATH "pytorch_512x512_f32.safetensors",
                              test_flags_t{
                                  .skip_cpu = 1
                              }),
            .group = test_group::f32,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_1024x1024_f32.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_f32, CONFIG_TEST_FILES_PATH "pytorch_1024x1024_f32.safetensors",
                              test_flags_t{
                                  .skip_cpu = 1
                              }),
            .group = test_group::f32,
        },
        {
            .name = "test_matrix_vs_pytorch(pytorch_2048x2048_f32.safetensors)",
            .func = std::bind(test_matrix_vs_pytorch_f32, CONFIG_TEST_FILES_PATH "pytorch_2048x2048_f32.safetensors",
                              test_flags_t{
                                  .skip_cpu = 1
                              }),
            .group = test_group::f32,
        },
    };

    std::vector<const test*> tests;
    tests.reserve(all_tests.size());
    for (const auto &test : all_tests) {
        if (test.group == test_group::i64 && !opt_enable_i64)
            continue;

        if (test.group == test_group::f32 && !opt_enable_f32)
            continue;

        tests.emplace_back(&test);
    }

    const u32 num_threads = [&] {
        if (opt_num_threads != 0)
            return opt_num_threads;

        long int num_cpus_online = sysconf(_SC_NPROCESSORS_ONLN);
        if (num_cpus_online <= 0)
            num_cpus_online = 4;

        const u32 num_threads = std::min(tests.size(), static_cast<size_t>(num_cpus_online));

        return num_threads;
    }();

    thread_pool threads(num_threads);

    threads.schedule([&](u32 thread_id) {
        const u32 batch_size = num_threads;
        u32 job_id = thread_id;

        while (job_id < tests.size()) {
            RUN_TEST(*tests[job_id]);
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

#include <mipc/file.h>
#include <rapidjson/document.h>

int run_gradient_descent()
{
    mat_f32_t xs, ypred, outw;

    /*
     * Parse safetensors file
     */
    auto ftensors = mipc::finbuf(CONFIG_GRAD_FILE_PATH);
    if (!ftensors) {
        fprintf(stderr, "Error: failed to open %s\n", CONFIG_GRAD_FILE_PATH);
        return 1;
    }

    const char *begin = ftensors.begin();

    u64 metadata_size = 0;
    std::memcpy(&metadata_size, begin, sizeof(u64));

    begin += sizeof(u64);

    std::string header_str;
    header_str.assign(begin, metadata_size+1);
    header_str[metadata_size] = '\0';

    rapidjson::Document header_json;
    header_json.Parse(header_str.c_str());
    assert(header_json.IsObject());

    struct safetensor_f32 {
        // Shape
        u32 rows = 0;
        u32 cols = 0;

        const f32* data = nullptr;
    };

    safetensor_f32 x, y;

    for (auto &m: header_json.GetObject()) {

        auto parse_tensor_data = [&] (safetensor_f32 &t) -> int {

            /*
             * Parse dtype
             */
            if (!m.value.HasMember("dtype")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "dtype");
                return 1;
            }

            if (!m.value["dtype"].IsString()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                        m.name.GetString(), "dtype", "string");
                return 1;
            }

            if (strcmp(m.value["dtype"].GetString(), "F32") != 0) {
                fmt::print(stderr, "{}: Expected {} field to be \"I32\", but got {}\n",
                        m.name.GetString(), "dtype", m.value.GetString());
                return 1;
            }

            /*
             * Parse shape
             */
            if (!m.value.HasMember("shape")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "shape");
                return 1;
            }

            if (!m.value["shape"].IsArray()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                       m.name.GetString(), "shape", "array");
                return 1;
            }

            if (m.value["shape"].GetArray().Size() != 2) {
                fmt::print(stderr, "{}: Expected shape to be array of size 2\n",
                       m.name.GetString());
                return 1;
            }

            t.rows = m.value["shape"].GetArray()[0].GetUint();
            t.cols = m.value["shape"].GetArray()[1].GetUint();


            /*
             * Parse data
             */
            if (!m.value.HasMember("data_offsets")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "data_offsets");
                return 1;
            }

            if (!m.value["data_offsets"].IsArray()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                        m.name.GetString(), "data_offsets", "array");
                return 1;
            }

            if (m.value["data_offsets"].GetArray().Size() != 2) {
                fmt::print(stderr, "{}: Expected data_offsets to be array of size 2\n",
                        m.name.GetString());
                return 1;
            }

            t.data = rcast<const f32*>(
                /* Get data offset as base. */
                m.value["data_offsets"].GetArray()[0].GetUint64() +

                /* Skip metadata size. */
                sizeof(u64) +

                /* Skip the actual metadata. */
                metadata_size +

                /* Finally, offset from base of mmaped file to obtain data pointer. */
                ftensors.begin()
            );

            return 0;
        };

        if (strcmp(m.name.GetString(), "x") == 0) {

            if (parse_tensor_data(x)) {
                fprintf(stderr, "Failed to parse %s\n", "x");
                return 1;
            }

            continue;
        }

        if (strcmp(m.name.GetString(), "y") == 0) {

            if (parse_tensor_data(y)) {
                fprintf(stderr, "Failed to parse %s\n", "y");
                return 1;
            }

            continue;
        }

        fprintf(stderr, "Expected \"x\" or \"y\", got: \"%s\"\n", m.name.GetString());
        return 1;
    }

    auto make_mat_from_tensor_data = [] (const safetensor_f32 &tensor) {
        return mat_f32_t::make_matrix_from_data(tensor.data, tensor.cols, tensor.rows);
    };

    auto mat_x = make_mat_from_tensor_data(x),
         mat_y = make_mat_from_tensor_data(y);

    mat_f32_t mat_w;
    f32 loss;

    puts("x");
    print_mat(mat_x);

    puts("y");
    print_mat(mat_y);

    run_kernel_cu_grad_f32(mat_x, mat_y, mat_w, loss);

    printf("data: %p, cols %u, rows %u, stride %u\n",
           mat_w.data.get(), mat_w.width, mat_w.height, mat_w.stride);

    puts("max_x:");
    print_mat(mat_x);

    puts("max_y:");
    print_mat(mat_y);

    printf("loss: %f\n", loss);

    return 0;
}

int run_classify()
{
    /*
     * Parse safetensors file
     */
    auto ftensors = mipc::finbuf(CONFIG_CLASSIFY_FILE_PATH);
    if (!ftensors) {
        fprintf(stderr, "Error: failed to open %s\n", CONFIG_GRAD_FILE_PATH);
        return 1;
    }

    const char *begin = ftensors.begin();

    u64 metadata_size = 0;
    std::memcpy(&metadata_size, begin, sizeof(u64));

    begin += sizeof(u64);

    std::string header_str;
    header_str.assign(begin, metadata_size+1);
    header_str[metadata_size] = '\0';

    rapidjson::Document header_json;
    header_json.Parse(header_str.c_str());
    assert(header_json.IsObject());

    struct safetensor_f32 {
        // Shape
        u32 rows = 0;
        u32 cols = 0;

        const f32* data = nullptr;
    };

    safetensor_f32 x, y;

    for (auto &m: header_json.GetObject()) {

        auto parse_tensor_data = [&] (safetensor_f32 &t) -> int {

            /*
             * Parse dtype
             */
            if (!m.value.HasMember("dtype")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "dtype");
                return 1;
            }

            if (!m.value["dtype"].IsString()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                        m.name.GetString(), "dtype", "string");
                return 1;
            }

            if (strcmp(m.value["dtype"].GetString(), "F32") != 0) {
                fmt::print(stderr, "{}: Expected {} field to be \"I32\", but got {}\n",
                        m.name.GetString(), "dtype", m.value.GetString());
                return 1;
            }

            /*
             * Parse shape
             */
            if (!m.value.HasMember("shape")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "shape");
                return 1;
            }

            if (!m.value["shape"].IsArray()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                       m.name.GetString(), "shape", "array");
                return 1;
            }

            if (m.value["shape"].GetArray().Size() == 0) {
                fmt::print(stderr, "{}: Expected shape to be non-zero\n",
                       m.name.GetString());
                return 1;
            }

            t.rows = m.value["shape"].GetArray()[0].GetUint();

            if (m.value["shape"].GetArray().Size() == 2)
                t.cols = m.value["shape"].GetArray()[1].GetUint();
            else
                t.cols = 1;


            /*
             * Parse data
             */
            if (!m.value.HasMember("data_offsets")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "data_offsets");
                return 1;
            }

            if (!m.value["data_offsets"].IsArray()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                        m.name.GetString(), "data_offsets", "array");
                return 1;
            }

            if (m.value["data_offsets"].GetArray().Size() != 2) {
                fmt::print(stderr, "{}: Expected data_offsets to be array of size 2\n",
                        m.name.GetString());
                return 1;
            }

            t.data = rcast<const f32*>(
                /* Get data offset as base. */
                m.value["data_offsets"].GetArray()[0].GetUint64() +

                /* Skip metadata size. */
                sizeof(u64) +

                /* Skip the actual metadata. */
                metadata_size +

                /* Finally, offset from base of mmaped file to obtain data pointer. */
                ftensors.begin()
            );

            return 0;
        };

        if (strcmp(m.name.GetString(), "x") == 0) {

            if (parse_tensor_data(x)) {
                fprintf(stderr, "Failed to parse %s\n", "x");
                return 1;
            }

            continue;
        }

        if (strcmp(m.name.GetString(), "y") == 0) {

            if (parse_tensor_data(y)) {
                fprintf(stderr, "Failed to parse %s\n", "y");
                return 1;
            }

            continue;
        }

        fprintf(stderr, "Expected \"x\" or \"y\", got: \"%s\"\n", m.name.GetString());
        return 1;
    }

    auto make_mat_from_tensor_data = [] (const safetensor_f32 &tensor) {
        return mat_f32_t::make_matrix_from_data(tensor.data, tensor.cols, tensor.rows);
    };

    auto mat_x = make_mat_from_tensor_data(x),
         mat_y = make_mat_from_tensor_data(y);

#if 0
    printf("xs: [%u;%u]\nys: [%u;%u]\n", mat_x.width, mat_x.height, mat_y.width, mat_y.height);
    puts("xs");
    print_mat(mat_x);

    puts("ygt");
    print_mat(mat_y);
#endif

    std::array<mat_f32_t, 3> hidden;
    f32 loss = NAN;

    train_cu_classify(mat_x, mat_y, hidden, loss);

    printf("loss: %.2f\n", loss);

    return 0;

}

int main(int argc, char **argv)
{
    int ret = 0;
    u32 num_threads = 0;
    bool explicit_enable = false;
    bool explicit_enable_f32 = false;
    bool explicit_enable_i64 = false;

    register_interrupt_handler();

    for (int arg = 1; arg < argc; ++arg) {
        const char *s = argv[arg];

        if (strcmp(s, "-h") == 0 || strcmp(s, "--help") == 0) {
            printf("Usage: %s OPTIONS\n", argv[0]);
            printf(
                "  -lc, --list-cuda    List available cuda devices\n"
                "       --bench        Print detailed branchmarking/timing information\n"
                "  -n,  --threads      Number of threads to run in parallel when running benchmarks\n"
                "       --test         Run test cuda kernel\n"
                "  -e,  --enable       Enable tests from group and run only them\n"
                "                        -ef32 | --enablef32 # Enables float32 tests\n"
                "                        -ei64 | --enablei64 # Enables int64 tests\n"
                "       --grad         Run only gradient descend test\n"
                "       --class        Run only classify test\n"
            );
            return 0;
        }

        if (strcmp(s, "-lc") == 0 || strcmp(s, "--list-cuda") == 0) {
            opt_list_cuda = true;
            continue;
        }

        if (strcmp(s, "--bench") == 0) {
            opt_bench = true;
            continue;
        }

        if (sscanf(s, "-n%u", &num_threads) == 1 || sscanf(s, "--threads%u", &num_threads)) {
            opt_num_threads = num_threads;
            continue;
        }

        if (strcmp(s, "--test") == 0) {
            opt_test = true;
            continue;
        }

        if (strcmp(s, "-ei64") == 0 || strcmp(s, "--enablei64") == 0) {
            explicit_enable     = true;
            explicit_enable_i64 = true;
            continue;
        }

        if (strcmp(s, "-ef32") == 0 || strcmp(s, "--enablef32") == 0) {
            explicit_enable     = true;
            explicit_enable_f32 = true;
            continue;
        }

        if (strcmp(s, "--grad") == 0) {
            opt_grad = true;
            continue;
        }

        if (strcmp(s, "--class") == 0) {
            opt_classify = true;
            continue;
        }
    }

    if (opt_grad)
        return run_gradient_descent();

    if (opt_classify)
        return run_classify();

    if (explicit_enable) {
        opt_enable_i64 = explicit_enable_i64;
        opt_enable_f32 = explicit_enable_f32;
    }

    matmul_cu_init(opt_list_cuda);
    if (opt_list_cuda)
        return 0;

    ret = run_tests();

    /* Print detailed benchmark/timing information */
    if (opt_bench) {
        auto binfo = benchinfo.consume_entries();

        if (!binfo.empty()) {
            printf("\nTensor source                       Kernel                    Time\n");
        }

        u32 even = 0;
        const char *clr;
        for (auto &[line, duration]: binfo) {
            append_time_string(line, duration, BENCHMARK_LINE_ALIGNMENT);

            clr = even ? CLR_WHITE : CLR_CYAN;
            even ^= 1;

            printf("%s%s", clr, line.c_str());
        }

        printf(CLR_RESET);
    }

    return ret;
}

