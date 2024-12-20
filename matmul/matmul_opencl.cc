#include <mutex>

#include <CL/cl.h>

#include <mipc/file.h>

#include "mat.h"
#include "config.h"

using mipc::finbuf;

std::mutex kctx_mtx;
struct cl_kernel_context
{
    bool initialized;
    cl_device_id device;
    cl_context context;
    cl_program program;
    finbuf kernel_source;

    cl_kernel_context() = default;

    ~cl_kernel_context()
    {
        std::unique_lock lck(kctx_mtx);

        if (!this->initialized)
            return;

        clReleaseProgram(this->program);
        clReleaseContext(this->context);
        this->initialized = false;
    }
} kctx;

static u32 cl_size_round(u32 size_bytes)
{
    constexpr u32 alignment_bytes = 64;
    return (size_bytes + alignment_bytes-1) & (~(alignment_bytes-1));
}

static int init_kernel_context_(struct cl_kernel_context &kctx)
{
    int err;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_program program;
    finbuf kernel_source;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err < 0) {
        fprintf(stderr, "clGetPlatformIDs: %d\n", err);
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err < 0) {
        fprintf(stderr, "clGetDeviceIDs: %d\n", err);
        return 1;
    }

    kernel_source = finbuf(CONFIG_CL_MATMUL_KERNEL_SRC);
    if (!kernel_source) {
        fprintf(stderr, "Failed to load kernel source\n");
        return 1;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateContext: %d\n", err);
        return 1;
    }

    const char*  sources[1]     = { kernel_source.data() };
    const size_t source_lens[1] = { kernel_source.size() };
    program = clCreateProgramWithSource(context, 1, sources, source_lens, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateProgramWithSource: %d\n", err);
        return 1;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        fprintf(stderr, "clBuildProgram: %d\n", err);
        return 1;
    }

    kctx.device = device;
    kctx.context = context;
    kctx.program = program;
    kctx.kernel_source = std::move(kernel_source);

    return 0;
}

static int init_kernel_context()
{
    int ret;

    std::unique_lock lck(kctx_mtx);

    if (kctx.initialized) [[likely]]
        return 0;

    ret = init_kernel_context_(kctx);

    if (ret)
        return ret;

    kctx.initialized = true;
    return 0;
}

static int run_kernel(matview_void_t lhs, matview_void_t rhs, matview_void_t out)
{
    int err;

    /* Actual work */
    cl_command_queue queue;
    cl_kernel kernel;
    cl_mem cl_out_buffer;
    cl_mem cl_lhs_buffer;
    cl_mem cl_rhs_buffer;
    u32 cl_out_buffer_size = cl_size_round(out.size_bytes());
    u32 cl_lhs_buffer_size = cl_size_round(lhs.size_bytes());
    u32 cl_rhs_buffer_size = cl_size_round(rhs.size_bytes());
    size_t local_size = 0, global_size = 0;

    const char * const kernel_name = [&]{
        assert(lhs.type == rhs.type);
        assert(lhs.type == out.type);

        switch(lhs.type) {
        case mat_type_e::i64:
            return "matmul_i64";
        case mat_type_e::f32:
            return "matmul_f32";
        }

        __builtin_unreachable();
    }();

    err = init_kernel_context();
    if (err)
        return err;

    cl_device_id device = kctx.device;
    cl_context context = kctx.context;
    cl_program program = kctx.program;

    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateCommandQueueWithProperties: %d\n", err);
        return 1;
    }

    kernel = clCreateKernel(program, kernel_name, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateKernel: %d\n", err);
        return 1;
    }

    cl_lhs_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, cl_lhs_buffer_size, NULL, &err);
    if (!cl_lhs_buffer) {
        fprintf(stderr, "Failed to allocate %s memory on the GPU: %d\n", "lhs", err);
        return 1;
    }

    cl_rhs_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_rhs_buffer_size, NULL, &err);
    if (!cl_rhs_buffer) {
        fprintf(stderr, "Failed to allocate %s memory on the GPU: %d\n", "rhs", err);
        return 1;
    }

    cl_out_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, cl_out_buffer_size, NULL, &err);
    if (!cl_out_buffer) {
        fprintf(stderr, "Failed to allocate %s memory on the GPU: %d\n", "out", err);
        return 1;
    }

    err  = clSetKernelArg(kernel, 0,  sizeof(cl_mem), &cl_lhs_buffer);
    err |= clSetKernelArg(kernel, 1,  sizeof(u32),    &lhs.width);
    err |= clSetKernelArg(kernel, 2,  sizeof(u32),    &lhs.height);
    err |= clSetKernelArg(kernel, 3,  sizeof(u32),    &lhs.stride);

    err |= clSetKernelArg(kernel, 4,  sizeof(cl_mem), &cl_rhs_buffer);
    err |= clSetKernelArg(kernel, 5,  sizeof(u32),    &rhs.width);
    err |= clSetKernelArg(kernel, 6,  sizeof(u32),    &rhs.height);
    err |= clSetKernelArg(kernel, 7,  sizeof(u32),    &rhs.stride);

    err |= clSetKernelArg(kernel, 8,  sizeof(cl_mem), &cl_out_buffer);
    err |= clSetKernelArg(kernel, 9,  sizeof(u32),    &out.width);
    err |= clSetKernelArg(kernel, 10, sizeof(u32),    &out.height);
    err |= clSetKernelArg(kernel, 11, sizeof(u32),    &out.stride);

    if (err < 0) {
        fprintf(stderr, "Failed to set kernel args\n");
        return 1;
    }

    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
    if (err < 0) {
        fprintf(stderr, "clGetKernelWorkGroupInfo: %d\n", err);
        return 1;
    }

    global_size = out.num_elems();
    if (local_size > global_size)
        local_size = global_size;

    err = clEnqueueWriteBuffer(queue, cl_lhs_buffer, CL_FALSE, 0, cl_lhs_buffer_size, lhs.data, 0, NULL, NULL);
    if (err < 0) {
        fprintf(stderr, "clEnqueueWriteBuffer %s: %d\n", "lhs", err);
        return 1;
    }

    err = clEnqueueWriteBuffer(queue, cl_rhs_buffer, CL_FALSE, 0, cl_rhs_buffer_size, rhs.data, 0, NULL, NULL);
    if (err < 0) {
        fprintf(stderr, "clEnqueueWriteBuffer %s: %d\n", "rhs", err);
        return 1;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err < 0) {
        fprintf(stderr, "clEnqueueNDRangeKernel %s: %d\n", "out", err);
        return 1;
    }

    clFinish(queue);

    err = clEnqueueReadBuffer(queue, cl_out_buffer, CL_TRUE, 0, cl_out_buffer_size, out.data, 0, NULL, NULL);
    if (err < 0) {
        fprintf(stderr, "clEnqueueReadBuffer: %d\n", err);
        return 1;
    }

    clFinish(queue);

    clReleaseMemObject(cl_out_buffer);
    clReleaseMemObject(cl_rhs_buffer);
    clReleaseMemObject(cl_lhs_buffer);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);

    return 0;
}

mat_i64_t mat_mul_cl(matview_i64_t lhs, matview_i64_t rhs)
{
    mat_i64_t ret = mat_i64_t::make_matrix_zero(lhs.height, rhs.width);

    run_kernel(lhs, rhs, ret);

    return ret;
}

mat_f32_t mat_mul_cl(matview_f32_t lhs, matview_f32_t rhs)
{
    mat_f32_t ret = mat_f32_t::make_matrix_zero(lhs.height, rhs.width);

    run_kernel(lhs, rhs, ret);

    return ret;
}
