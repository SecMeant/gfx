__kernel void matmul_i64(
    __global ulong* lhs,
    uint lhs_cols,
    uint lhs_rows,
    uint lhs_stride,

    __global ulong* rhs,
    uint rhs_cols,
    uint rhs_rows,
    uint rhs_stride,

    __global ulong* out,
    uint out_cols,
    uint out_rows,
    uint out_stride
) {
    const uint thread_id = (uint)get_global_id(0);
    const uint y = thread_id / out_cols;
    const uint x = thread_id % out_cols;

    if (y >= out_rows)
        return;

    out[x + y * out_stride] = 0;

    /* We assert: lhs_cols == rhs_rows */

    for (uint i = 0; i < lhs_cols; ++i)
        out[x + y * out_stride] += lhs[i + lhs_stride * y] * rhs[x + rhs_stride * i];
}

__kernel void matmul_f32(
    __global float* lhs,
    uint lhs_cols,
    uint lhs_rows,
    uint lhs_stride,

    __global float* rhs,
    uint rhs_cols,
    uint rhs_rows,
    uint rhs_stride,

    __global float* out,
    uint out_cols,
    uint out_rows,
    uint out_stride
) {
    const uint thread_id = (uint)get_global_id(0);
    const uint y = thread_id / out_cols;
    const uint x = thread_id % out_cols;

    if (y >= out_rows)
        return;

    out[x + y * out_stride] = 0;

    /* We assert: lhs_cols == rhs_rows */

    for (uint i = 0; i < lhs_cols; ++i)
        out[x + y * out_stride] += lhs[i + lhs_stride * y] * rhs[x + rhs_stride * i];
}
