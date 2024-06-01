#pragma once

#include <exception>
#include <string>

struct {
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

#define RUN_TEST(test) \
    do { \
        try { \
            ++test_stats.num_tests; \
            test(); \
        } catch (const std::exception &e) { \
            printf(STR(test) ": %s\n", e.what()); \
            ++test_stats.num_failed; \
        } \
    } while(0)


