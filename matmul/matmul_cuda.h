#pragma once

#include "types.h"

#if defined(__cplusplus)
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

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
    u32  h_out_stride
);

