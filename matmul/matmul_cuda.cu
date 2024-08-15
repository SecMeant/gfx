#include <stdio.h>
#include <assert.h>

#include <vector>
#include <mutex>

#include <cuda_runtime.h>

#include "ansi_codes.h"
#include "types.h"
#include "matmul_cuda.h"

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
    assert(lhs_cols == lhs_rows);
    assert(rhs_cols == rhs_rows);
    assert(out_cols == out_rows);
    assert(lhs_cols == rhs_cols);
    assert(lhs_cols == out_cols);

    if (strcmp(current_dev().name, "NVIDIA GeForce GTX 970")) {
        fprintf(stderr, "WARN: %s: kernel written for %s, but current device is %s.",
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
    }

    __builtin_unreachable();
}

