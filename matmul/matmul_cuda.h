#pragma once

#include "mat.h"
#include "types.h"

#if defined(__cplusplus)
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

enum class cuda_kernel_variant {
    UMEM,
    UMEM_TILED,
    TILED,
    TEST,
};

EXTERN_C int matmul_cu_init(bool verbose);

EXTERN_C int run_kernel_cu(
    i64 *h_lhs,
    u32  h_lhs_cols,
    u32  h_lhs_rows,
    u32  h_lhs_stride,

    i64 *h_rhs,
    u32  h_rhs_cols,
    u32  h_rhs_rows,
    u32  h_rhs_stride,

    i64 *h_out,
    u32  h_out_cols,
    u32  h_out_rows,
    u32  h_out_stride,

    cuda_kernel_variant variant
);

EXTERN_C int run_kernel_cu_f32(
    f32 *h_lhs,
    u32  h_lhs_cols,
    u32  h_lhs_rows,
    u32  h_lhs_stride,

    f32 *h_rhs,
    u32  h_rhs_cols,
    u32  h_rhs_rows,
    u32  h_rhs_stride,

    f32 *h_out,
    u32  h_out_cols,
    u32  h_out_rows,
    u32  h_out_stride,

    cuda_kernel_variant variant
);

EXTERN_C void run_kernel_cu_grad_f32(matview_f32_t h_x, matview_f32_t h_ygt, mat_f32_t &h_out, f32 &out_loss);
EXTERN_C void train_cu_classify(matview_f32_t h_x, matview_f32_t h_ygt, std::array<mat_f32_t, 3> &h_out, f32 &out_loss);

