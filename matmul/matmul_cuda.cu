/* vim: set tabstop=4 softtabstop=4 expandtab shiftwidth=4: */
#include <stdio.h>
#include <assert.h>

#include <atomic>
#include <vector>
#include <mutex>

#include <cuda_runtime.h>

#include "ansi_codes.h"
#include "types.h"
#include "matmul_cuda.h"

#include "print_utils.h"

#define checkCudaError(err) \
({ \
    __typeof__(err) _err = (err); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorName(_err)); \
        exit(EXIT_FAILURE); \
    } \
})


/*
 * Guard against executing multiple CUDA streams at once.
 *
 * FIXME: For some reason we crash when we run from multiple threads.
 *        Figure out how to use the API concurrently.
 */
static std::mutex kernel_exec_mtx;

static std::vector<cudaDeviceProp> cuda_devices;
static const cudaDeviceProp& current_dev()
{
    assert(!cuda_devices.empty());
    return cuda_devices[0];
}

EXTERN_C int matmul_cu_init(bool verbose)
{
    int num_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_devices);

    checkCudaError(err);

    if (num_devices <= 0) {
        printf("No cuda devices\n");
        return 1;
    }

    if (verbose)
        printf("num devices: %d\n", num_devices);

    for (int dev_id = 0; dev_id < num_devices; ++dev_id) {
        auto &devprop = cuda_devices.emplace_back();
        checkCudaError(cudaGetDeviceProperties(&devprop, dev_id));

        if (!verbose)
            continue;

        printf("name                : %s\n", devprop.name);
        printf("  processor count   : %d\n", devprop.multiProcessorCount);
        printf("  max blocks/proc   : %d\n", devprop.maxBlocksPerMultiProcessor);
        printf("  max threads/block : %d\n", devprop.maxThreadsPerBlock);
        printf("  max threads X     : %d\n", devprop.maxThreadsDim[0]);
        printf("  max threads Y     : %d\n", devprop.maxThreadsDim[1]);
        printf("  max threads Z     : %d\n", devprop.maxThreadsDim[2]);
        printf("  max grid X        : %d\n", devprop.maxGridSize[0]);
        printf("  max grid Y        : %d\n", devprop.maxGridSize[1]);
        printf("  max grid Z        : %d\n", devprop.maxGridSize[2]);
        printf("  warp size         : %d\n", devprop.warpSize);
        printf("  global memory     : %lu MB\n", devprop.totalGlobalMem / (1024UL * 1024UL));
        printf("  shared memory     : %lu KB\n", devprop.sharedMemPerBlock / 1024UL);
        printf("  L2                : %d KB\n", devprop.l2CacheSize / 1024);
        printf("  gpu clock         : %d MHz\n", devprop.clockRate / 1024);
        printf("  mem clock         : %d MHz\n", devprop.memoryClockRate / 1024);
        printf("  bus width         : %d\n", devprop.memoryBusWidth);
        putchar('\n');
    }

    return 0;
}


/****************************************************************************
 * I64 Kernels
 ****************************************************************************/

/*
 * We assume all matrcies are square and have the same dimensions.
 */
__global__ void kernel_matmul_cu(
    const i64 *lhs,
    const i64 *rhs,
          i64 *out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
        const u32 x = threadIdx.x + blockDim.x * blockIdx.x;
        const u32 y = threadIdx.y + blockDim.y * blockIdx.y;

        const bool out_of_bounds = x >= dim | y >= dim;
        if (out_of_bounds)
            return;

        out[x + y*out_stride] = 0;

        for (u32 i = 0; i < dim; ++i)
            out[x + y*out_stride] += lhs[i + y*lhs_stride] * rhs[x + i*rhs_stride];
}

/*
 * We assume all matrcies are square and have the same dimensions.
 */
__global__ void kernel_matmul_tiled_cu(
    const i64 *lhs,
    const i64 *rhs,
          i64 *out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
    constexpr u32 TILE_SIZE = 32 * 32;

    __shared__ i64 tilea[TILE_SIZE];
    __shared__ i64 tileb[TILE_SIZE];
    __shared__ i64 tilec[TILE_SIZE];

    /* Global coordinates for this thread. */
    const u32 gx = threadIdx.x + blockIdx.x * blockDim.x;
    const u32 gy = threadIdx.y + blockIdx.y * blockDim.y;

    const bool out_of_bounds = gx >= dim | gy >= dim;
    if (out_of_bounds)
        return;

    /* Local tile coordinates for this thread. */
    const u32 tx      = threadIdx.x;
    const u32 ty      = threadIdx.y;
    const u32 tstride = blockDim.x;

    tilec[tx + ty * tstride] = 0;

    /*
     * Row is a loop invariant for tilea.
     * Column is a loop invariant for tileb.
     */
    for (int j = 0; j < dim; j += blockDim.x) {

        /*
         * Load lhs and rhs tiles into shared memory.
         * Each thread loads one element.
         */
        tilea[tx + ty * tstride] = lhs[(tx+j) + gy * lhs_stride];
        tileb[tx + ty * tstride] = rhs[gx + (ty+j) * rhs_stride];

        /* Make sure tilea and tileb are populated. */
        __syncthreads();

        /* Accumulate results for current tile. */
        for (u32 i = 0; i < blockDim.x; ++i)
            tilec[tx + ty * tstride] += tilea[i + ty * tstride] * tileb[tx + i * tstride];

        /* Make sure we don't modify tilea and tileb before all threads are finihsed with per tile computation */
        __syncthreads();
    }

    out[gx + gy * out_stride] = tilec[tx + ty * tstride];
}

/*
 * We assume all matrcies are square and have the same dimensions.
 */
__global__ void kernel_matmul_test_cu(
    const i64 *lhs,
    const i64 *rhs,
          i64 *out_,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
    /* Global coordinates for this thread. */
    const u32 gx = blockIdx.z + blockIdx.x * blockDim.x;
    const u32 gy = blockIdx.z + blockIdx.y * blockDim.y;

    const bool out_of_bounds = gx >= dim | gy >= dim;
    if (out_of_bounds)
        return;

    u32 out = 0;

    for (u32 i = 0; i < dim; ++i)
        out += lhs[i + gy * lhs_stride] * rhs[gx + i * rhs_stride];

    out_[gx + gy * out_stride] = out;
}

static int run_kernel_cu_umem_(
    const i64 *h_lhs,
    const i64 *h_rhs,
          i64 *h_out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
          i64 *u_lhs;
    const u32  lhs_size = dim * lhs_stride * sizeof(*u_lhs);

          i64 *u_rhs;
    const u32  rhs_size = dim * rhs_stride * sizeof(*u_rhs);

          i64 *u_out;
    const u32  out_size = dim * out_stride * sizeof(*u_out);

    checkCudaError(cudaMallocManaged(&u_lhs, lhs_size));
    checkCudaError(cudaMallocManaged(&u_rhs, rhs_size));
    checkCudaError(cudaMallocManaged(&u_out, out_size));

    memcpy(u_lhs, h_lhs, lhs_size);
    memcpy(u_rhs, h_rhs, rhs_size);

    /* We assume block dim to be 32 */
    const u32 num_threads = std::min(dim, 32u);
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = (dim + 31u) / 32u;
    const dim3 grid_dims(num_blocks, num_blocks);

    kernel_matmul_cu<<<grid_dims, block_dims>>>(
        u_lhs,
        u_rhs,
        u_out,
        dim,
        lhs_stride,
        rhs_stride,
        out_stride
    );
    cudaDeviceSynchronize();

    memcpy(h_out, u_out, out_size);

    cudaFree(u_out);
    cudaFree(u_rhs);
    cudaFree(u_lhs);

    checkCudaError(cudaGetLastError());

    return 0;
}

static int run_kernel_cu_umem_tiled_(
    const i64 *h_lhs,
    const i64 *h_rhs,
          i64 *h_out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
          i64 *u_lhs;
    const u32  lhs_size = dim * lhs_stride * sizeof(*u_lhs);

          i64 *u_rhs;
    const u32  rhs_size = dim * rhs_stride * sizeof(*u_rhs);

          i64 *u_out;
    const u32  out_size = dim * out_stride * sizeof(*u_out);

    checkCudaError(cudaMallocManaged(&u_lhs, lhs_size));
    checkCudaError(cudaMallocManaged(&u_rhs, rhs_size));
    checkCudaError(cudaMallocManaged(&u_out, out_size));

    memcpy(u_lhs, h_lhs, lhs_size);
    memcpy(u_rhs, h_rhs, rhs_size);

    /* We assume block dim to be 32 */
    const u32 num_threads = std::min(dim, 32u);
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = (dim + num_threads - 1) / num_threads;
    const dim3 grid_dims(num_blocks, num_blocks);

    kernel_matmul_tiled_cu<<<grid_dims, block_dims>>>(
        u_lhs,
        u_rhs,
        u_out,
        dim,
        lhs_stride,
        rhs_stride,
        out_stride
    );
    cudaDeviceSynchronize();

    memcpy(h_out, u_out, out_size);

    cudaFree(u_out);
    cudaFree(u_rhs);
    cudaFree(u_lhs);

    checkCudaError(cudaGetLastError());

    return 0;
}

static int run_kernel_cu_tiled_(
    const i64 *h_lhs,
    const i64 *h_rhs,
          i64 *h_out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
          i64 *d_lhs;
    const u32  lhs_size = dim * lhs_stride * sizeof(*d_lhs);

          i64 *d_rhs;
    const u32  rhs_size = dim * rhs_stride * sizeof(*d_rhs);

          i64 *d_out;
    const u32  out_size = dim * out_stride * sizeof(*d_out);

    checkCudaError(cudaMalloc(&d_lhs, lhs_size));
    checkCudaError(cudaMalloc(&d_rhs, rhs_size));
    checkCudaError(cudaMalloc(&d_out, out_size));

    checkCudaError(cudaMemcpy(d_lhs, h_lhs, lhs_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_rhs, h_rhs, rhs_size, cudaMemcpyHostToDevice));

    /* We assume block dim to be 32 */
    const u32 num_threads = std::min(dim, 32u);
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = (dim + num_threads - 1) / num_threads;
    const dim3 grid_dims(num_blocks, num_blocks);

    kernel_matmul_tiled_cu<<<grid_dims, block_dims>>>(
        d_lhs,
        d_rhs,
        d_out,
        dim,
        lhs_stride,
        rhs_stride,
        out_stride
    );
    cudaDeviceSynchronize();

    checkCudaError(cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost));

    cudaFree(d_out);
    cudaFree(d_rhs);
    cudaFree(d_lhs);

    checkCudaError(cudaGetLastError());

    return 0;
}

static int run_kernel_cu_test_(
    const i64 *h_lhs,
    const i64 *h_rhs,
          i64 *h_out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
          i64 *u_lhs;
    const u32  lhs_size = dim * lhs_stride * sizeof(*u_lhs);

          i64 *u_rhs;
    const u32  rhs_size = dim * rhs_stride * sizeof(*u_rhs);

          i64 *u_out;
    const u32  out_size = dim * out_stride * sizeof(*u_out);

    checkCudaError(cudaMallocManaged(&u_lhs, lhs_size));
    checkCudaError(cudaMallocManaged(&u_rhs, rhs_size));
    checkCudaError(cudaMallocManaged(&u_out, out_size));

    memcpy(u_lhs, h_lhs, lhs_size);
    memcpy(u_rhs, h_rhs, rhs_size);

    const u32 num_threads = 32;
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = (dim + num_threads - 1) / num_threads;
    const dim3 grid_dims(num_blocks, num_blocks);

    kernel_matmul_cu<<<grid_dims, block_dims>>>(
        u_lhs,
        u_rhs,
        u_out,
        dim,
        lhs_stride,
        rhs_stride,
        out_stride
    );
    cudaDeviceSynchronize();

    memcpy(h_out, u_out, out_size);

    cudaFree(u_out);
    cudaFree(u_rhs);
    cudaFree(u_lhs);

    checkCudaError(cudaGetLastError());

    return 0;
}

EXTERN_C int run_kernel_cu(
          i64 *h_lhs,
    const u32  lhs_cols,
    const u32  lhs_rows,
    const u32  lhs_stride,

          i64 *h_rhs,
    const u32  rhs_cols,
    const u32  rhs_rows,
    const u32  rhs_stride,

          i64 *h_out,
    const u32  out_cols,
    const u32  out_rows,
    const u32  out_stride,

    cuda_kernel_variant variant
) {
    static std::atomic<u32> printed(0);

    assert(lhs_cols == lhs_rows);
    assert(rhs_cols == rhs_rows);
    assert(out_cols == out_rows);
    assert(lhs_cols == rhs_cols);
    assert(lhs_cols == out_cols);

    if (printed.fetch_or(1, std::memory_order_relaxed) == 0 &&
        strcmp(current_dev().name, "NVIDIA GeForce GTX 970")) {
        fprintf(stderr, CLR_YELLOW "WARN: %s: kernel written for %s, but current device is %s.\n" CLR_RESET,
                __func__, "NVIDIA GeForce GTX 970", current_dev().name);
    }

    std::unique_lock lck(kernel_exec_mtx);

    switch (variant) {
    case cuda_kernel_variant::UMEM:
        return run_kernel_cu_umem_(
            h_lhs,
            h_rhs,
            h_out,
            lhs_cols,
            lhs_stride,
            rhs_stride,
            out_stride
        );

    case cuda_kernel_variant::UMEM_TILED:
        return run_kernel_cu_umem_tiled_(
            h_lhs,
            h_rhs,
            h_out,
            lhs_cols,
            lhs_stride,
            rhs_stride,
            out_stride
        );

    case cuda_kernel_variant::TILED:
        return run_kernel_cu_tiled_(
            h_lhs,
            h_rhs,
            h_out,
            lhs_cols,
            lhs_stride,
            rhs_stride,
            out_stride
        );

    case cuda_kernel_variant::TEST:
        return run_kernel_cu_test_(
            h_lhs,
            h_rhs,
            h_out,
            lhs_cols,
            lhs_stride,
            rhs_stride,
            out_stride
        );
    }

    __builtin_unreachable();
}


/****************************************************************************
 * F32 Kernels
 ****************************************************************************/

/*
 * We assume all matrcies are square and have the same dimensions.
 */
__global__ void kernel_matmul_cu_f32(
    const f32 *lhs,
    const f32 *rhs,
          f32 *out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
        const u32 x = threadIdx.x + blockDim.x * blockIdx.x;
        const u32 y = threadIdx.y + blockDim.y * blockIdx.y;

        const bool out_of_bounds = x >= dim | y >= dim;
        if (out_of_bounds)
            return;

        out[x + y*out_stride] = 0;

        for (u32 i = 0; i < dim; ++i)
            out[x + y*out_stride] += lhs[i + y*lhs_stride] * rhs[x + i*rhs_stride];
}

/*
 * We assume all matrcies are square and have the same dimensions.
 */
__global__ void kernel_matmul_tiled_cu_f32(
    const f32 *lhs,
    const f32 *rhs,
          f32 *out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
    constexpr u32 TILE_SIZE = 32 * 32;

    __shared__ f32 tilea[TILE_SIZE];
    __shared__ f32 tileb[TILE_SIZE];
    __shared__ f32 tilec[TILE_SIZE];

    /* Global coordinates for this thread. */
    const u32 gx = threadIdx.x + blockIdx.x * blockDim.x;
    const u32 gy = threadIdx.y + blockIdx.y * blockDim.y;

    const bool out_of_bounds = gx >= dim | gy >= dim;
    if (out_of_bounds)
        return;

    /* Local tile coordinates for this thread. */
    const u32 tx      = threadIdx.x;
    const u32 ty      = threadIdx.y;
    const u32 tstride = blockDim.x;

    tilec[tx + ty * tstride] = 0;

    /*
     * Row is a loop invariant for tilea.
     * Column is a loop invariant for tileb.
     */
    for (int j = 0; j < dim; j += blockDim.x) {

        /*
         * Load lhs and rhs tiles into shared memory.
         * Each thread loads one element.
         */
        tilea[tx + ty * tstride] = lhs[(tx+j) + gy * lhs_stride];
        tileb[tx + ty * tstride] = rhs[gx + (ty+j) * rhs_stride];

        /* Make sure tilea and tileb are populated. */
        __syncthreads();

        /* Accumulate results for current tile. */
        for (u32 i = 0; i < blockDim.x; ++i)
            tilec[tx + ty * tstride] += tilea[i + ty * tstride] * tileb[tx + i * tstride];

        /* Make sure we don't modify tilea and tileb before all threads are finihsed with per tile computation */
        __syncthreads();
    }

    out[gx + gy * out_stride] = tilec[tx + ty * tstride];
}

/*
 * We assume all matrcies are square and have the same dimensions.
 */
__global__ void kernel_matmul_test_cu_f32(
    const f32 *lhs,
    const f32 *rhs,
          f32 *out_,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
    /* Global coordinates for this thread. */
    const u32 gx = blockIdx.z + blockIdx.x * blockDim.x;
    const u32 gy = blockIdx.z + blockIdx.y * blockDim.y;

    const bool out_of_bounds = gx >= dim | gy >= dim;
    if (out_of_bounds)
        return;

    u32 out = 0;

    for (u32 i = 0; i < dim; ++i)
        out += lhs[i + gy * lhs_stride] * rhs[gx + i * rhs_stride];

    out_[gx + gy * out_stride] = out;
}

static int run_kernel_cu_umem_f32_(
    const f32 *h_lhs,
    const f32 *h_rhs,
          f32 *h_out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
          f32 *u_lhs;
    const u32  lhs_size = dim * lhs_stride * sizeof(*u_lhs);

          f32 *u_rhs;
    const u32  rhs_size = dim * rhs_stride * sizeof(*u_rhs);

          f32 *u_out;
    const u32  out_size = dim * out_stride * sizeof(*u_out);

    checkCudaError(cudaMallocManaged(&u_lhs, lhs_size));
    checkCudaError(cudaMallocManaged(&u_rhs, rhs_size));
    checkCudaError(cudaMallocManaged(&u_out, out_size));

    memcpy(u_lhs, h_lhs, lhs_size);
    memcpy(u_rhs, h_rhs, rhs_size);

    /* We assume block dim to be 32 */
    const u32 num_threads = std::min(dim, 32u);
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = (dim + 31u) / 32u;
    const dim3 grid_dims(num_blocks, num_blocks);

    kernel_matmul_cu_f32<<<grid_dims, block_dims>>>(
        u_lhs,
        u_rhs,
        u_out,
        dim,
        lhs_stride,
        rhs_stride,
        out_stride
    );
    cudaDeviceSynchronize();

    memcpy(h_out, u_out, out_size);

    cudaFree(u_out);
    cudaFree(u_rhs);
    cudaFree(u_lhs);

    checkCudaError(cudaGetLastError());

    return 0;
}

static int run_kernel_cu_umem_tiled_f32_(
    const f32 *h_lhs,
    const f32 *h_rhs,
          f32 *h_out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
          f32 *u_lhs;
    const u32  lhs_size = dim * lhs_stride * sizeof(*u_lhs);

          f32 *u_rhs;
    const u32  rhs_size = dim * rhs_stride * sizeof(*u_rhs);

          f32 *u_out;
    const u32  out_size = dim * out_stride * sizeof(*u_out);

    checkCudaError(cudaMallocManaged(&u_lhs, lhs_size));
    checkCudaError(cudaMallocManaged(&u_rhs, rhs_size));
    checkCudaError(cudaMallocManaged(&u_out, out_size));

    memcpy(u_lhs, h_lhs, lhs_size);
    memcpy(u_rhs, h_rhs, rhs_size);

    /* We assume block dim to be 32 */
    const u32 num_threads = std::min(dim, 32u);
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = (dim + num_threads - 1) / num_threads;
    const dim3 grid_dims(num_blocks, num_blocks);

    kernel_matmul_tiled_cu_f32<<<grid_dims, block_dims>>>(
        u_lhs,
        u_rhs,
        u_out,
        dim,
        lhs_stride,
        rhs_stride,
        out_stride
    );
    cudaDeviceSynchronize();

    memcpy(h_out, u_out, out_size);

    cudaFree(u_out);
    cudaFree(u_rhs);
    cudaFree(u_lhs);

    checkCudaError(cudaGetLastError());

    return 0;
}

static int run_kernel_cu_tiled_f32_(
    const f32 *h_lhs,
    const f32 *h_rhs,
          f32 *h_out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
          f32 *d_lhs;
    const u32  lhs_size = dim * lhs_stride * sizeof(*d_lhs);

          f32 *d_rhs;
    const u32  rhs_size = dim * rhs_stride * sizeof(*d_rhs);

          f32 *d_out;
    const u32  out_size = dim * out_stride * sizeof(*d_out);

    checkCudaError(cudaMalloc(&d_lhs, lhs_size));
    checkCudaError(cudaMalloc(&d_rhs, rhs_size));
    checkCudaError(cudaMalloc(&d_out, out_size));

    checkCudaError(cudaMemcpy(d_lhs, h_lhs, lhs_size, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_rhs, h_rhs, rhs_size, cudaMemcpyHostToDevice));

    /* We assume block dim to be 32 */
    const u32 num_threads = std::min(dim, 32u);
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = (dim + num_threads - 1) / num_threads;
    const dim3 grid_dims(num_blocks, num_blocks);

    kernel_matmul_tiled_cu_f32<<<grid_dims, block_dims>>>(
        d_lhs,
        d_rhs,
        d_out,
        dim,
        lhs_stride,
        rhs_stride,
        out_stride
    );
    cudaDeviceSynchronize();

    checkCudaError(cudaMemcpy(h_out, d_out, out_size, cudaMemcpyDeviceToHost));

    cudaFree(d_out);
    cudaFree(d_rhs);
    cudaFree(d_lhs);

    checkCudaError(cudaGetLastError());

    return 0;
}

static int run_kernel_cu_test_f32_(
    const f32 *h_lhs,
    const f32 *h_rhs,
          f32 *h_out,
    const u32  dim,
    const u32  lhs_stride,
    const u32  rhs_stride,
    const u32  out_stride
) {
          f32 *u_lhs;
    const u32  lhs_size = dim * lhs_stride * sizeof(*u_lhs);

          f32 *u_rhs;
    const u32  rhs_size = dim * rhs_stride * sizeof(*u_rhs);

          f32 *u_out;
    const u32  out_size = dim * out_stride * sizeof(*u_out);

    checkCudaError(cudaMallocManaged(&u_lhs, lhs_size));
    checkCudaError(cudaMallocManaged(&u_rhs, rhs_size));
    checkCudaError(cudaMallocManaged(&u_out, out_size));

    memcpy(u_lhs, h_lhs, lhs_size);
    memcpy(u_rhs, h_rhs, rhs_size);

    const u32 num_threads = 32;
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = (dim + num_threads - 1) / num_threads;
    const dim3 grid_dims(num_blocks, num_blocks);

    kernel_matmul_cu_f32<<<grid_dims, block_dims>>>(
        u_lhs,
        u_rhs,
        u_out,
        dim,
        lhs_stride,
        rhs_stride,
        out_stride
    );
    cudaDeviceSynchronize();

    memcpy(h_out, u_out, out_size);

    cudaFree(u_out);
    cudaFree(u_rhs);
    cudaFree(u_lhs);

    checkCudaError(cudaGetLastError());

    return 0;
}

EXTERN_C int run_kernel_cu_f32(
          f32 *h_lhs,
    const u32  lhs_cols,
    const u32  lhs_rows,
    const u32  lhs_stride,

          f32 *h_rhs,
    const u32  rhs_cols,
    const u32  rhs_rows,
    const u32  rhs_stride,

          f32 *h_out,
    const u32  out_cols,
    const u32  out_rows,
    const u32  out_stride,

    cuda_kernel_variant variant
) {
    static std::atomic<u32> printed(0);

    assert(lhs_cols == lhs_rows);
    assert(rhs_cols == rhs_rows);
    assert(out_cols == out_rows);
    assert(lhs_cols == rhs_cols);
    assert(lhs_cols == out_cols);

    if (printed.fetch_or(1, std::memory_order_relaxed) == 0 &&
        strcmp(current_dev().name, "NVIDIA GeForce GTX 970")) {
        fprintf(stderr, CLR_YELLOW "WARN: %s: kernel written for %s, but current device is %s.\n" CLR_RESET,
                __func__, "NVIDIA GeForce GTX 970", current_dev().name);
    }

    std::unique_lock lck(kernel_exec_mtx);

    switch (variant) {
    case cuda_kernel_variant::UMEM:
        return run_kernel_cu_umem_f32_(
            h_lhs,
            h_rhs,
            h_out,
            lhs_cols,
            lhs_stride,
            rhs_stride,
            out_stride
        );

    case cuda_kernel_variant::UMEM_TILED:
        return run_kernel_cu_umem_tiled_f32_(
            h_lhs,
            h_rhs,
            h_out,
            lhs_cols,
            lhs_stride,
            rhs_stride,
            out_stride
        );

    case cuda_kernel_variant::TILED:
        return run_kernel_cu_tiled_f32_(
            h_lhs,
            h_rhs,
            h_out,
            lhs_cols,
            lhs_stride,
            rhs_stride,
            out_stride
        );

    case cuda_kernel_variant::TEST:
        return run_kernel_cu_test_f32_(
            h_lhs,
            h_rhs,
            h_out,
            lhs_cols,
            lhs_stride,
            rhs_stride,
            out_stride
        );
    }

    __builtin_unreachable();
}

struct matrix {
    constexpr static u32 STRIDE_AUTO = 0;

    enum class grad_t {
        no_grad,
        with_grad,
    };

    void free()
    {
        if (this->data)
            cudaFree(this->data);

        if (this->grad)
            cudaFree(this->grad);

        this->data = nullptr;
        this->grad = nullptr;
        this->width = 0;
        this->height = 0;
        this->stride = 0;
    }

    cudaError_t alloc_on_device(u32 width, u32 height, u32 stride, grad_t grad_opt)
    {
        this->free();

        if (stride == STRIDE_AUTO)
            stride = width;

        assert(width <= stride);

        auto err = cudaMalloc(&this->data, sizeof(*this->data) * height * stride);
        if (err != cudaSuccess)
            return err;

        if (grad_opt == grad_t::with_grad) {
            err = cudaMalloc(&this->grad, sizeof(*this->grad) * height * stride);
            if (err != cudaSuccess) {
                cudaFree(this->data);
                this->data = nullptr;
                return err;
            }
        }

        this->width = width;
        this->height = height;
        this->stride = stride;

        return cudaSuccess;
    }

    u64 size_bytes() const
    {
        return this->height * this->stride * sizeof(*this->data);
    }

    f32 *data = nullptr;
    f32 *grad = nullptr;
    u32  width = 0;
    u32  height = 0;
    u32  stride = 0;
};

constexpr static auto STRIDE_AUTO = matrix::STRIDE_AUTO;

constexpr static auto no_grad   = matrix::grad_t::no_grad;
constexpr static auto with_grad = matrix::grad_t::with_grad;

/*
 * Computes:
 *     out = lhs @ rhs
 *
 *  Assumes 'lhs', 'rhs' and 'out' are correctly sized.
 */
__global__ void kernel_matmul(
    struct matrix lhs,
    struct matrix rhs,
    struct matrix out
) {
    const u32 gx = threadIdx.x + blockIdx.x * blockDim.x;
    const u32 gy = threadIdx.y + blockIdx.y * blockDim.y;

    const bool out_of_bounds = (gx >= out.width) | (gy >= out.height);
    if (out_of_bounds)
        return;

    f32 v = 0;
    for (u32 i = 0; i < lhs.width; ++i)
        v += lhs.data[i + gy * lhs.stride] * rhs.data[gx + i * rhs.stride];
    out.data[gx + gy * out.stride] = v;
}

__global__ void kernel_mse(
    struct matrix lhs,
    struct matrix rhs,
    struct matrix out_mse,
    f32 *out_mse_v
){
    const u32 gx = threadIdx.x + blockIdx.x * blockDim.x;
    const u32 gy = threadIdx.y + blockIdx.y * blockDim.y;

    const bool out_of_bounds = (gx >= out_mse.width) | (gy >= out_mse.height);
    if (out_of_bounds)
        return;

    /*
     * Compute:
     *     data = (lhs - rhs) ** 2
     *     grad = 2 * (lhs - rhs)
     */
    f32 mse = lhs.data[gx + gy * lhs.stride] - rhs.data[gx + gy * rhs.stride];
    out_mse.grad[gx + gy * out_mse.stride] = mse;
    out_mse.data[gx + gy * out_mse.stride] = mse * mse;

    __syncthreads();

    /*
     * From now on we execute:
     *     num_vectors = lhs.width = rhs.width
     *     x: [0; num_vectors]
     *     y: 0
     *
     *  IOW reduce across columns.
     */
    if (gy != 0)
        return;

    f32 mse_v = 0;
    for (u32 y = 0; y < out_mse.height; ++y)
        mse_v += out_mse.data[gx + y * out_mse.stride];

    *out_mse_v = mse_v;
}

__global__ void kernel_grad_cu_forward(
    struct matrix xs,
    struct matrix ys,
    struct matrix ypred,
    struct matrix w,
    struct matrix loss0,
    struct matrix loss1,
    f32 *out_loss_v
){
    const u32 gx = threadIdx.x;
    const u32 gy = threadIdx.y;

    const bool out_of_bounds = (gx >= ypred.width) | (gy >= ypred.height);

    if (out_of_bounds)
        return;

    __syncthreads();

    if (gx != 0 | gy != 0)
        return;

    f32 loss_v = 0;
    for (u32 i = 0; i < loss1.height; ++i)
        loss_v += loss1.data[gx + i * loss1.stride];

    *out_loss_v = loss_v;
}

__global__ void kernel_grad_reset(struct matrix m)
{
    const u32 gx = threadIdx.x;
    const u32 gy = threadIdx.y;

    const bool out_of_bounds = (gx >= m.width) | (gy >= m.height);
    if (out_of_bounds)
        return;

    m.grad[gx + gy * m.stride] = 1.0f;
}

__global__ void kernel_backward_x_squared(
    struct matrix cur,
    struct matrix prev
) {
    const u32 gx = threadIdx.x;
    const u32 gy = threadIdx.y;

    const bool out_of_bounds = (gx >= cur.width) | (gy >= cur.height);
    if (out_of_bounds)
        return;

    prev.grad[gx + gy * prev.stride] *= 2*cur.grad[gx + gy * cur.stride];
}

__global__ void kernel_grad_cu_backward(
    struct matrix xs,
    struct matrix w,
    struct matrix loss
){
    const f32 lr = 1e-6f;

    const u32 gx = threadIdx.x;
    const u32 gy = threadIdx.y;

    const bool out_of_bounds = (gx >= w.width) | (gy >= w.height);
    if (out_of_bounds)
        return;

    w.grad[gx + gy * w.stride] = 2*loss.grad[0 + gy * loss.stride] * xs.data[0 + gx * xs.stride];
    w.data[gx + gy * w.stride] -= w.grad[gx + gy * w.stride] * lr;
}

EXTERN_C void run_kernel_cu_grad_f32(
    matview_f32_t h_xs,
    matview_f32_t h_ygt,
    mat_f32_t &h_weights,
    f32 &out_loss
) {
    assert(h_xs.width == h_ygt.width);
    assert(h_xs.height == h_ygt.height);
    assert(h_xs.width == 1);
    assert(h_ygt.width == 1);

    struct matrix d_xs, d_ygt, d_ypred, d_weights, d_loss;
    f32 *d_lossv;
    const u32 num_neurons = h_xs.height;

    checkCudaError(d_xs.alloc_on_device(h_xs.width, h_xs.height, h_xs.stride, no_grad));
    checkCudaError(d_ygt.alloc_on_device(d_ygt.width, h_ygt.height, h_ygt.stride, no_grad));
    checkCudaError(d_ypred.alloc_on_device(h_ygt.width, h_ygt.height, h_ygt.stride, no_grad));
    checkCudaError(d_weights.alloc_on_device(h_xs.height, num_neurons, STRIDE_AUTO, with_grad));
    checkCudaError(d_loss.alloc_on_device(d_ypred.width, d_ypred.height, d_ypred.stride, with_grad));
    checkCudaError(cudaMalloc(&d_lossv, sizeof(*d_lossv)));

    mat_f32_t h_grad = mat_f32_t::make_matrix_zero(d_weights.width, d_weights.height, d_weights.stride);
    h_weights = mat_f32_t::make_matrix_zero(d_weights.width, d_weights.height, d_weights.stride);

    assert(d_xs.stride == h_xs.stride);
    assert(d_xs.size_bytes() == h_xs.size_bytes());
    checkCudaError(cudaMemcpy(d_xs.data, h_xs.data, d_xs.size_bytes(), cudaMemcpyHostToDevice));

    assert(d_ygt.stride == h_ygt.stride);
    assert(d_ygt.size_bytes() == h_ygt.size_bytes());
    checkCudaError(cudaMemcpy(d_ygt.data, h_ygt.data, d_ygt.size_bytes(), cudaMemcpyHostToDevice));

    assert(d_weights.size_bytes() == h_weights.size_bytes());
    checkCudaError(cudaMemcpy(d_weights.data, h_weights.data.get(), d_weights.size_bytes(), cudaMemcpyHostToDevice));


    const u32 num_threads = 32;
    const dim3 block_dims(num_threads, num_threads);

    const u32 num_blocks = 1;
    const dim3 grid_dims(num_blocks, num_blocks);

    constexpr f32 LOSS_TARGET = 1e-5f;
    u32 epoch = 0;
    while(1) {

        /*
         * Forward pass
         */
        kernel_matmul<<<grid_dims, block_dims>>>(
            d_weights,
            d_xs,
            d_ypred
        );

        kernel_mse<<<grid_dims, block_dims>>>(
            d_ypred,
            d_ygt,
            d_loss,
            d_lossv
        );

        checkCudaError(cudaMemcpy(&out_loss, d_lossv, sizeof(f32), cudaMemcpyDeviceToHost));

        kernel_grad_cu_backward<<<grid_dims, block_dims>>>(
            d_xs,
            d_weights,
            d_loss
        );

        checkCudaError(cudaMemcpy(h_grad.data.get(), d_weights.grad, d_weights.size_bytes(),
                                  cudaMemcpyDeviceToHost));

        /* Copy trained weights back to the caller. */
        checkCudaError(cudaMemcpy(h_weights.data.get(), d_weights.data, d_weights.size_bytes(),
                                  cudaMemcpyDeviceToHost));

        /* Copy loss back to the caller. */
        checkCudaError(cudaMemcpy(&out_loss, d_lossv, sizeof(f32),
                                  cudaMemcpyDeviceToHost));

        if (++epoch % 1024 == 0 || out_loss <= LOSS_TARGET) {
            puts("grad");
            print_mat(h_grad);

            puts("weights");
            print_mat(h_weights);

            auto ypred = mat_mul_cpu(h_weights, h_xs);
            puts("ypred");
            print_mat(ypred);

            printf("loss: %f\n", out_loss);
        }

        if (out_loss <= LOSS_TARGET)
            break;
    }

    cudaFree(d_lossv);
    d_loss.free();
    d_weights.free();
    d_ypred.free();
    d_ygt.free();
    d_xs.free();
}

