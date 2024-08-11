#include "matmul_cuda.h"
#include "mat.h"

static mat_t mat_mul_cu_(matview_t lhs, matview_t rhs, cuda_kernel_variant variant)
{
    assert(lhs.width == lhs.height);
    assert(rhs.width == rhs.height);
    assert(lhs.width == rhs.width);

    mat_t out = mat_t::make_matrix(lhs.width, lhs.height);

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

mat_t mat_mul_cu(matview_t lhs, matview_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::UMEM);
}

mat_t mat_mul_cu_tiled(matview_t lhs, matview_t rhs)
{
    return mat_mul_cu_(lhs, rhs, cuda_kernel_variant::UMEM_TILED);
}
