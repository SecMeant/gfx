#pragma once

#include <exception>
#include <string>

#include "mat.h"
#include "ansi_codes.h"

struct [[maybe_unused]] {
    size_t num_tests;
    size_t num_failed;
} inline test_stats;

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

#define RUN_TEST(test, ...) \
    do { \
        printf(STR(test) ": ..."); \
        fflush(stdout); \
        try { \
            ++test_stats.num_tests; \
            test(__VA_ARGS__); \
            printf("\b \b\b\b" CLR_GREEN "OK\n" CLR_RESET); \
        } catch (const std::exception &e) { \
            printf("\b\b\b" CLR_RED "Failed: %s\n" CLR_RESET, e.what()); \
            ++test_stats.num_failed; \
        } \
        fflush(stdout); \
    } while(0)

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


