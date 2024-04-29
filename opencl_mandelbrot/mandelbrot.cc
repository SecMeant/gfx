#include <stdio.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include <CL/cl.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "types.h"
#include "config.h"

/* FIXME: Don't assume from where we are running. */
#if CONFIG_CL_COMPILE_ONLINE
static const char SPIRV_FILEPATH[] = "./mandelbrot.cl";
#else
static const char SPIRV_FILEPATH[] = "./mandelbrot.spv";
#endif

static constexpr u32 IMAGE_WIDTH = 3840;
static constexpr u32 IMAGE_HEIGHT = 2160;
static constexpr u32 IMAGE_BYTES_PER_PIXEL = 4;
static constexpr u32 IMAGE_SIZE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_BYTES_PER_PIXEL;

#include "render.cc"

static int
load_spirv(const char* const path, char** spirv_, size_t* spirv_size_)
{
    struct stat spirv_stat;
    char* spirv = NULL;
    int fd_spirv;

    fd_spirv = open(path, O_RDONLY);
    if (fd_spirv < 0) {
        perror("open(spirv)");
        return 1;
    }

    if (fstat(fd_spirv, &spirv_stat)) {
        perror("fstat");
        // Leak the file descriptor, we are going to exit anyway
        return 1;
    }

    spirv = static_cast<char*>(malloc(spirv_stat.st_size));
    if (read(fd_spirv, spirv, spirv_stat.st_size) != spirv_stat.st_size) {
        // This shouldn't happen unless the file is huge and read() bailed out
        // early and want us to read in chunks or someone is writing to the
        // file as we read it. In the first case, we just don't care for now. For the second case,
        // it's just a weird state, don't even try to fallback, exit with failure.
        //
        // Also leak the file descriptorm, we are going to exit anyway.
        fprintf(stderr, "Sanity check failed: amount of bytes read\n");
        return 1;
    }

    *spirv_ = spirv;
    *spirv_size_ = spirv_stat.st_size;

    return 0;
}

static void
describe_build_error()
{
    // TODO ;]
    fprintf(stderr, "shit happened\n");
}

static int
bitmap_render_cl(const u32 bitmap_width, const u32 bitmap_height, u8* const bitmap)
{
    int err;

    /* For OpenCL init itself */
    cl_platform_id platform;
    cl_device_id device;

    /* For loading SPIRV into the GPU */
    cl_context context;
    cl_program program;
    size_t spirv_size = 0;
    char* spirv = NULL;

    /* Actual work */
    cl_command_queue queue;
    cl_kernel kernel;
    cl_mem output_buffer;
    size_t local_size = 0, global_size = 0;
    const u32 bitmap_size_bytes = bitmap_width * bitmap_height * IMAGE_BYTES_PER_PIXEL;

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

    err = load_spirv(SPIRV_FILEPATH, &spirv, &spirv_size);
    if (err) {
        fprintf(stderr, "Failed to load spirv: %d\n", err);
        return 1;
    }

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateContext: %d\n", err);
        return 1;
    }

#if CONFIG_CL_COMPILE_ONLINE
    const char*  sources[1]     = { spirv };
    const size_t source_lens[1] = { spirv_size };
    program = clCreateProgramWithSource(context, 1, sources, source_lens, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateProgramWithSource: %d\n", err);
        return 1;
    }
#else
    program = clCreateProgramWithIL(context, spirv, spirv_size, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateProgramWithIL: %d\n", err);
        return 1;
    }
#endif

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0) {
        describe_build_error();
        return 1;
    }

    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    if (err < 0) {
        fprintf(stderr, "clCreateCommandQueueWithProperties: %d\n", err);
        return 1;
    }

    kernel = clCreateKernel(program, "mandelbrot", &err);
    if (err < 0) {
        fprintf(stderr, "clCreateKernel: %d\n", err);
        return 1;
    }

    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bitmap_size_bytes, NULL, &err);
    if (!output_buffer) {
        fprintf(stderr, "Failed to allocate memory on the GPU: %d\n", err);
        return 1;
    }

    err = clSetKernelArg(kernel, 0, sizeof(u32), &bitmap_width);
    err |= clSetKernelArg(kernel, 1, sizeof(u32), &bitmap_height);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_buffer);
    if (err < 0) {
        fprintf(stderr, "Failed to set kernel args\n");
        return 1;
    }

    err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
    if (err < 0) {
        fprintf(stderr, "clGetKernelWorkGroupInfo: %d\n", err);
        return 1;
    }

    global_size = bitmap_width * bitmap_height;
    if (local_size > global_size)
        local_size = global_size;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err < 0) {
        fprintf(stderr, "clEnqueueNDRangeKernel: %d\n", err);
        return 1;
    }

    clFinish(queue);

    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, bitmap_size_bytes, bitmap, 0, NULL, NULL);
    if (err < 0) {
        fprintf(stderr, "clEnqueueReadBuffer: %d\n", err);
        return 1;
    }

    clReleaseKernel(kernel);
    clReleaseMemObject(output_buffer);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    return 0;
}

int
main()
{
    /*
     * Prepare the buffers
     */
    std::vector<u8> bitmap_data(IMAGE_SIZE_BYTES, 0);

    /*
     * Render the image in memory
     */
    bitmap_render_cl(IMAGE_WIDTH, IMAGE_HEIGHT, std::data(bitmap_data));

    /*
     * Draw the rendered bitmap on screen or save to the file
     */
    bitmap_save(std::data(bitmap_data), IMAGE_WIDTH, IMAGE_HEIGHT);

    return 0;
}

