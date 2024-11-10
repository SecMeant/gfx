#include "mat.h"
#include "types.h"

#include <cassert>

/*
 * Common implementations for matrix operations on CPU
 */

template <typename MatrixType, typename ViewType>
MatrixType mat_add_cpu_(ViewType lhs, ViewType rhs)
{
    assert(lhs.width == rhs.width);
    assert(lhs.height == rhs.height);

    MatrixType out = MatrixType::make_matrix_zero(lhs.width, lhs.height, lhs.stride);

    for (u32 y = 0; y < lhs.height; ++y)
        for (u32 x = 0; x < lhs.width; ++x)
            out[x,y] = lhs[x,y] + rhs[x,y];

    return out;
}

template <typename MatrixType, typename ViewType>
MatrixType mat_sub_cpu_(ViewType lhs, ViewType rhs)
{
    assert(lhs.width == rhs.width);
    assert(lhs.height == rhs.height);

    MatrixType out = MatrixType::make_matrix_zero(lhs.width, lhs.height, lhs.stride);

    for (u32 y = 0; y < lhs.height; ++y)
        for (u32 x = 0; x < lhs.width; ++x)
            out[x,y] = lhs[x,y] - rhs[x,y];

    return out;
}

template <typename MatrixType, typename ViewType>
MatrixType mat_mul_cpu_(ViewType lhs, ViewType rhs)
{
    assert(lhs.width == rhs.height);
    assert(lhs.height == rhs.width);

    MatrixType out = MatrixType::make_matrix_zero(lhs.height, rhs.width);

    for (u32 y = 0; y < lhs.height; ++y)
        for (u32 x = 0; x < lhs.width; ++x)
            for (u32 i = 0; i < lhs.width; ++i)
                out[x,y] += lhs[i,y] * rhs[x,i];

    return out;
}

template <typename ViewType>
void mat_copy_(ViewType dst, ViewType src)
{
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    const auto height = dst.height;
    const auto width = dst.width;

    for (u32 y = 0; y < height; ++y)
        for (u32 x = 0; x < width; ++x)
            dst[x, y] = src[x, y];
}

mat_i64_t mat_add_cpu(matview_i64_t lhs, matview_i64_t rhs)
{ return mat_add_cpu_<mat_i64_t, matview_i64_t>(lhs, rhs); }

mat_i64_t mat_sub_cpu(matview_i64_t lhs, matview_i64_t rhs)
{ return mat_sub_cpu_<mat_i64_t, matview_i64_t>(lhs, rhs); }

mat_i64_t mat_mul_cpu(matview_i64_t lhs, matview_i64_t rhs)
{ return mat_mul_cpu_<mat_i64_t, matview_i64_t>(lhs, rhs); }

void mat_copy(matview_i64_t dst, matview_i64_t src)
{ mat_copy_<matview_i64_t>(dst, src); }

mat_f32_t mat_add_cpu(matview_f32_t lhs, matview_f32_t rhs)
{ return mat_add_cpu_<mat_f32_t, matview_f32_t>(lhs, rhs); }

mat_f32_t mat_sub_cpu(matview_f32_t lhs, matview_f32_t rhs)
{ return mat_sub_cpu_<mat_f32_t, matview_f32_t>(lhs, rhs); }

mat_f32_t mat_mul_cpu(matview_f32_t lhs, matview_f32_t rhs)
{ return mat_mul_cpu_<mat_f32_t, matview_f32_t>(lhs, rhs); }

void mat_copy(matview_f32_t dst, matview_f32_t src)
{ mat_copy_<matview_f32_t>(dst, src); }

template <typename ViewType>
static void assert_mat_square(ViewType m)
{
    assert(m.width == m.height);
}

template <typename ViewType>
static void assert_mat_mullable(ViewType lhs, ViewType rhs)
{
    /* We don't support non square for now. */
    assert_mat_square<ViewType>(lhs);
    assert_mat_square<ViewType>(rhs);
    assert(lhs.width == rhs.width);
}

template <typename ViewType, typename MatrixType = ViewType::ParentType>
MatrixType strassen_cpu_small_(ViewType lhs, ViewType rhs, MatrixType out)
{
    assert_mat_square<ViewType>(lhs);
    assert_mat_square<ViewType>(rhs);
    assert_mat_square<ViewType>(out);
    assert(lhs.width == rhs.width);
    assert(lhs.width == out.width);

    mat_copy(out, mat_mul_cpu(lhs, rhs));

    return out;
}

template <typename ViewType, typename MatrixType = ViewType::ParentType>
MatrixType strassen_cpu_common(ViewType lhs, ViewType rhs)
{
    assert_mat_mullable<ViewType>(lhs, rhs);

    MatrixType out = MatrixType::make_matrix_zero(lhs.width, lhs.height, lhs.stride);

    if (lhs.width <= 4)
        return strassen_cpu_small_<ViewType>(lhs, rhs, std::move(out));

    assert(lhs.width % 4 == 0);
    const auto quarter_size = lhs.width / 2;

    ViewType a11(&lhs[0,0], quarter_size, quarter_size, lhs.stride);
    ViewType a12(&lhs[quarter_size,0], quarter_size, quarter_size, lhs.stride);
    ViewType a21(&lhs[0,quarter_size], quarter_size, quarter_size, lhs.stride);
    ViewType a22(&lhs[quarter_size,quarter_size], quarter_size, quarter_size, lhs.stride);

    ViewType b11(&rhs[0,0], quarter_size, quarter_size, rhs.stride);
    ViewType b12(&rhs[quarter_size,0], quarter_size, quarter_size, rhs.stride);
    ViewType b21(&rhs[0,quarter_size], quarter_size, quarter_size, rhs.stride);
    ViewType b22(&rhs[quarter_size,quarter_size], quarter_size, quarter_size, rhs.stride);

    MatrixType m1 = strassen_cpu(mat_add_cpu(a11, a22), mat_add_cpu(b11, b22));
    MatrixType m2 = strassen_cpu(mat_add_cpu(a21, a22), b11);
    MatrixType m3 = strassen_cpu(a11, mat_sub_cpu(b12, b22));
    MatrixType m4 = strassen_cpu(a22, mat_sub_cpu(b21, b11));
    MatrixType m5 = strassen_cpu(mat_add_cpu(a11, a12), b22);
    MatrixType m6 = strassen_cpu(mat_sub_cpu(a21, a11), mat_add_cpu(b11, b12));
    MatrixType m7 = strassen_cpu(mat_sub_cpu(a12, a22), mat_add_cpu(b21, b22));

    ViewType c11(&out[0,0], quarter_size, quarter_size, out.stride);
    ViewType c12(&out[quarter_size,0], quarter_size, quarter_size, out.stride);
    ViewType c21(&out[0,quarter_size], quarter_size, quarter_size, out.stride);
    ViewType c22(&out[quarter_size,quarter_size], quarter_size, quarter_size, out.stride);

    mat_copy(c11, mat_add_cpu(mat_sub_cpu(mat_add_cpu(m1, m4), m5), m7));
    mat_copy(c12, mat_add_cpu(m3, m5));
    mat_copy(c21, mat_add_cpu(m2, m4));
    mat_copy(c22, mat_add_cpu(mat_add_cpu(mat_sub_cpu(m1, m2), m3), m6));

    return out;
}

mat_i64_t strassen_cpu(matview_i64_t lhs, matview_i64_t rhs)
{ return strassen_cpu_common(lhs, rhs); }

mat_f32_t strassen_cpu(matview_f32_t lhs, matview_f32_t rhs)
{ return strassen_cpu_common(lhs, rhs); }

