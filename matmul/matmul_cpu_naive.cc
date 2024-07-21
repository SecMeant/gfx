#include "mat.h"
#include "types.h"

#include <cassert>

mat_t mat_add_cpu(matview_t lhs, matview_t rhs)
{
    assert(lhs.width == rhs.width);
    assert(lhs.height == rhs.height);

    mat_t out = mat_t::make_matrix_zero(lhs.width, lhs.height, lhs.stride);

    for (u32 y = 0; y < lhs.height; ++y)
        for (u32 x = 0; x < lhs.width; ++x)
            out[x,y] = lhs[x,y] + rhs[x,y];

    return out;
}

mat_t mat_sub_cpu(matview_t lhs, matview_t rhs)
{
    assert(lhs.width == rhs.width);
    assert(lhs.height == rhs.height);

    mat_t out = mat_t::make_matrix_zero(lhs.width, lhs.height, lhs.stride);

    for (u32 y = 0; y < lhs.height; ++y)
        for (u32 x = 0; x < lhs.width; ++x)
            out[x,y] = lhs[x,y] - rhs[x,y];

    return out;
}

mat_t mat_mul_cpu(matview_t lhs, matview_t rhs)
{
    assert(lhs.width == rhs.height);
    assert(lhs.height == rhs.width);

    mat_t out = mat_t::make_matrix_zero(lhs.height, rhs.width);

    for (u32 y = 0; y < lhs.height; ++y)
        for (u32 x = 0; x < lhs.width; ++x)
            for (u32 i = 0; i < lhs.width; ++i)
                out[x,y] += lhs[i,y] * rhs[x,i];

    return out;
}

void mat_copy(matview_t dst, matview_t src)
{
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    const auto height = dst.height;
    const auto width = dst.width;

    for (u32 y = 0; y < height; ++y)
        for (u32 x = 0; x < width; ++x)
            dst[x, y] = src[x, y];
}

static void assert_mat_square(matview_t m)
{
    assert(m.width == m.height);
}

static void assert_mat_mullable(matview_t lhs, matview_t rhs)
{
    /* We don't support non square for now. */
    assert_mat_square(lhs);
    assert_mat_square(rhs);
    assert(lhs.width == rhs.width);
}

static mat_t strassen_cpu_small_(matview_t lhs, matview_t rhs, mat_t out)
{
    assert_mat_square(lhs);
    assert_mat_square(rhs);
    assert_mat_square(out);
    assert(lhs.width == rhs.width);
    assert(lhs.width == out.width);

    mat_copy(out, mat_mul_cpu(lhs, rhs));

    return out;
}

mat_t strassen_cpu(matview_t lhs, matview_t rhs)
{
    assert_mat_mullable(lhs, rhs);

    mat_t out = mat_t::make_matrix_zero(lhs.width, lhs.height, lhs.stride);

    if (lhs.width <= 4)
        return strassen_cpu_small_(lhs, rhs, std::move(out));

    assert(lhs.width % 4 == 0);
    const auto quarter_size = lhs.width / 2;

    matview_t a11(&lhs[0,0], quarter_size, quarter_size, lhs.stride);
    matview_t a12(&lhs[quarter_size,0], quarter_size, quarter_size, lhs.stride);
    matview_t a21(&lhs[0,quarter_size], quarter_size, quarter_size, lhs.stride);
    matview_t a22(&lhs[quarter_size,quarter_size], quarter_size, quarter_size, lhs.stride);

    matview_t b11(&rhs[0,0], quarter_size, quarter_size, rhs.stride);
    matview_t b12(&rhs[quarter_size,0], quarter_size, quarter_size, rhs.stride);
    matview_t b21(&rhs[0,quarter_size], quarter_size, quarter_size, rhs.stride);
    matview_t b22(&rhs[quarter_size,quarter_size], quarter_size, quarter_size, rhs.stride);

    mat_t m1 = strassen_cpu(mat_add_cpu(a11, a22), mat_add_cpu(b11, b22));
    mat_t m2 = strassen_cpu(mat_add_cpu(a21, a22), b11);
    mat_t m3 = strassen_cpu(a11, mat_sub_cpu(b12, b22));
    mat_t m4 = strassen_cpu(a22, mat_sub_cpu(b21, b11));
    mat_t m5 = strassen_cpu(mat_add_cpu(a11, a12), b22);
    mat_t m6 = strassen_cpu(mat_sub_cpu(a21, a11), mat_add_cpu(b11, b12));
    mat_t m7 = strassen_cpu(mat_sub_cpu(a12, a22), mat_add_cpu(b21, b22));

    matview_t c11(&out[0,0], quarter_size, quarter_size, out.stride);
    matview_t c12(&out[quarter_size,0], quarter_size, quarter_size, out.stride);
    matview_t c21(&out[0,quarter_size], quarter_size, quarter_size, out.stride);
    matview_t c22(&out[quarter_size,quarter_size], quarter_size, quarter_size, out.stride);

    mat_copy(c11, mat_add_cpu(mat_sub_cpu(mat_add_cpu(m1, m4), m5), m7));
    mat_copy(c12, mat_add_cpu(m3, m5));
    mat_copy(c21, mat_add_cpu(m2, m4));
    mat_copy(c22, mat_add_cpu(mat_add_cpu(mat_sub_cpu(m1, m2), m3), m6));

    return out;
}
