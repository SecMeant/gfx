#include "test.h"
#include "mat.h"
#include "print_utils.h"

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

int main()
{
    RUN_TEST(test_matrix_simple_add);
    RUN_TEST(test_matrix_simple_mul);
    RUN_TEST(test_matrix_simple_strassen_mul);

    printf("Tests run   : %zu\nTests failed: %zu\n",
           test_stats.num_tests, test_stats.num_failed);

    return 0;
}
