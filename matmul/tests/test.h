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

struct test {
    std::string name;
    std::function<void()> func;
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
    matview_t actual,
    matview_t expected,
    matview_t lhs = matview_t(),
    matview_t rhs = matview_t(),
    mat_op op = mat_op::none
);


