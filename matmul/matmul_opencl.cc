#include <CL/cl.h>

#include <mipc/file.h>

#include "mat.h"
#include "config.h"

static u32 cl_size_round(u32 size_bytes)
{
    constexpr u32 alignment_bytes = 64;
    return (size_bytes + alignment_bytes-1) & (~(alignment_bytes-1));
}

static int run_kernel(matview_t lhs, matview_t rhs, mat_t &out_)
{
    using mipc::finbuf;

    int err;

    /* For OpenCL init itself */
    cl_platform_id platform;
    cl_device_id device;

    /* For loading SPIRV into the GPU */
    cl_context context;
    cl_program program;
    finbuf kernel_source;

    /* Actual work */
    mat_t out = mat_t::make_matrix_zero(lhs.height, rhs.width);
    cl_command_queue queue;
    cl_kernel kernel;
    cl_mem cl_out_buffer;
    cl_mem cl_lhs_buffer;
    cl_mem cl_rhs_buffer;
    u32 cl_out_buffer_size = cl_size_round(out.size_bytes());
    u32 cl_lhs_buffer_size = cl_size_round(lhs.size_bytes());
    u32 cl_rhs_buffer_size = cl_size_round(rhs.size_bytes());
    size_t local_size = 0, global_size = 0;

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

    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateCommandQueueWithProperties: %d\n", err);
        return 1;
    }

    kernel = clCreateKernel(program, "matmul", &err);
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

    err = clEnqueueReadBuffer(queue, cl_out_buffer, CL_TRUE, 0, cl_out_buffer_size, out.data.get(), 0, NULL, NULL);
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
    clReleaseProgram(program);
    clReleaseContext(context);

    out_ = std::move(out);

    return 0;
}

mat_t mat_add_cl(matview_t lhs, matview_t rhs)
{
    return mat_t();
}

mat_t mat_sub_cl(matview_t lhs, matview_t rhs)
{
    return mat_t();
}

mat_t mat_mul_cl(matview_t lhs, matview_t rhs)
{
    mat_t ret;

    run_kernel(lhs, rhs, ret);

    return ret;
}
