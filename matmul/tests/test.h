#pragma once

#include <atomic>
#include <exception>
#include <string>
#include <functional>

#include <fmt/format.h>

#include "mat.h"
#include "../ansi_codes.h"

struct [[maybe_unused]] {
    std::atomic_uint64_t num_tests = 0;
    std::atomic_uint64_t num_failed = 0;
} inline test_stats;

enum class test_group {
    common = 0,
    i64,
    f32,
};

struct test {
    std::string name;
    std::function<void()> func;
    test_group group = test_group::common;
};

struct test_flags_t {
    u32 skip_cpu: 1 = 0;
};

#define STR_(x) #x
#define STR(x) STR_(x)

#define TEST_ASSERT(expr) \
    do { \
        if (!(expr)) { \
            throw test_failure("Assertion failed: " #expr " in file " __FILE__ ", line " STR(__LINE__)); \
        } \
    } while (0)

struct test_failure : public std::exception {
    explicit test_failure(std::string msg)
    : message(std::move(msg))
    {}

    const char* what() const noexcept override
    {
        return message.c_str();
    }

    std::string message;
};

enum mat_op {
    none,
    mul,
};

/*
 * Compares matrices, actual and expected.
 * If they miscompare, prints at which offset the miscompare happened.
 *
 * Optionally takes lhs, rhs matrices and operation that produced the result.
 * They are used to print more context, useful when debugging.
 */
void mat_compare_or_fail(
    const char *test_name,
    matview_i64_t actual,
    matview_i64_t expected,
    matview_i64_t lhs = matview_i64_t(),
    matview_i64_t rhs = matview_i64_t(),
    mat_op op = mat_op::none
);


