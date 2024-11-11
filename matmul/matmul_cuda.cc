#include "matmul_cuda.h"
#include "mat.h"

static mat_i64_t mat_mul_cu_(matview_i64_t lhs, matview_i64_t rhs, cuda_kernel_variant variant)
{
    assert(lhs.width == lhs.height);
    assert(rhs.width == rhs.height);
    assert(lhs.width == rhs.width);

    mat_i64_t out = mat_i64_t::make_matrix(lhs.width, lhs.height);

    run_kernel_cu(
        lhs.data,
        lhs.width,
        lhs.height,
        lhs.stride,

        rhs.data,
        rhs.width,
        rhs.height,
        rhs.stride,

        out.data.get(),
        out.width,
        out.height,
        out.stride,

        variant
    );

    return out;
}

mat_i64_t mat_mul_cu(matview_i64_t lhs, matview_i64_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::UMEM);
}

mat_i64_t mat_mul_cu_umem_tiled(matview_i64_t lhs, matview_i64_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::UMEM_TILED);
}

mat_i64_t mat_mul_cu_tiled(matview_i64_t lhs, matview_i64_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::TILED);
}

mat_i64_t mat_mul_cu_test(matview_i64_t lhs, matview_i64_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::TEST);
}

static mat_f32_t mat_mul_cu_(matview_f32_t lhs, matview_f32_t rhs, cuda_kernel_variant variant)
{
    assert(lhs.width == lhs.height);
    assert(rhs.width == rhs.height);
    assert(lhs.width == rhs.width);

    mat_f32_t out = mat_f32_t::make_matrix(lhs.width, lhs.height);

    run_kernel_cu_f32(
        lhs.data,
        lhs.width,
        lhs.height,
        lhs.stride,

        rhs.data,
        rhs.width,
        rhs.height,
        rhs.stride,

        out.data.get(),
        out.width,
        out.height,
        out.stride,

        variant
    );

    return out;
}

mat_f32_t mat_mul_cu(matview_f32_t lhs, matview_f32_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::UMEM);
}

mat_f32_t mat_mul_cu_umem_tiled(matview_f32_t lhs, matview_f32_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::UMEM_TILED);
}

mat_f32_t mat_mul_cu_tiled(matview_f32_t lhs, matview_f32_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::TILED);
}

mat_f32_t mat_mul_cu_test(matview_f32_t lhs, matview_f32_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::TEST);
}
