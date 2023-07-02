#include <stdio.h>

#include <CL/cl.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

// FIXME: don't hardcode the path...
#define SPIRV_FILEPATH "/home/holz/opencl_oldschool/bruteforce.bc"

int load_spirv(const char *path, void **spirv_, size_t *spirv_size_)
{
	struct stat spirv_stat;
	void *spirv = NULL;
	int fd_spirv = open(path, O_RDONLY);

	if (fd_spirv < 0) {
		perror("open(spirv)");
		return -1;
	}

	if (fstat(fd_spirv, &spirv_stat)) {
		perror("open");
		// Leak the file descriptor, we are going to exit anyway
		return -1;
	}

	spirv = malloc(spirv_stat.st_size);
	if (read(fd_spirv, spirv, spirv_stat.st_size) != spirv_stat.st_size) {
		// This shouldn't happen unless some is writing to the file as we read it.
		// It's weird state, don't even try to fallback, just exit with failure.
		fprintf(stderr, "Sanity check failed: amount of bytes read\n");
		return -1;
	}

	*spirv_ = spirv;
	*spirv_size_ = spirv_stat.st_size;
	return 0;
}

static void describe_build_error()
{
	// TODO ;]
	fprintf(stderr, "shit happened\n");
}

static void set_random(float *data, size_t size)
{
	srand(1337);
	for (size_t i = 0; i < size; ++i) {
		data[i] = rand() / (float)RAND_MAX;
	}
}

int main(void)
{
	int err;

	/* For OpenCL init itself */
	cl_platform_id platform;
	cl_device_id device;

	/* For loading SPIRV into the GPU */
	cl_context context;
	cl_program program;
	size_t spirv_size = 0;
	void *spirv = NULL;

	/* Actual work */
	float input[1024], output[1024];
	const size_t count = 1024;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_mem input_buffer, output_buffer;
	size_t local_size = 0, global_size = 0;

	err = clGetPlatformIDs(1, &platform, NULL);
	if (err < 0) {
		perror("clGetPlatformIDs");
		return 1;
	}

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err < 0) {
		perror("clGetDeviceIDs");
		return 1;
	}

	if (load_spirv(SPIRV_FILEPATH, &spirv, &spirv_size)) {
		fprintf(stderr, "Failed to load spirv\n");
		return 1;
	}

	context = clCreateContext(NULL, 1, &device, NULL , NULL, &err);
	if (err < 0) {
		perror("clCreateContext");
		return 1;
	}

	program = clCreateProgramWithIL(context, spirv, spirv_size, &err);
	if (err < 0) {
		perror("clCreateProgramWithIL");
		return 1;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err < 0) {
		describe_build_error();
		return 1;
	}

	queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
	if (err < 0) {
		perror("clCreateCommandQueueWithProperties\n");
		return 1;
	}

	kernel = clCreateKernel(program, "square", &err);
	if (err < 0) {
		perror("clCreateKernel\n");
		return 1;
	}

	set_random(input, count);

	input_buffer  = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(input), NULL, NULL);
	output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(output), NULL, NULL);
	if (!input_buffer || !output_buffer) {
		fprintf(stderr, "Failed to allocate memory on the GPU\n");
		return 1;
	}

	err = clEnqueueWriteBuffer(queue, input_buffer, CL_TRUE, 0, sizeof(input), input, 0, NULL, NULL);
	if (err < 0) {
		perror("clEnqueueWriteBuffer");
		return 1;
	}

	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
	err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
	if (err < 0) {
		fprintf(stderr, "Failed to set kernel args\n");
		return 1;
	}

	err = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL);
	if (err < 0) {
		perror("clGetKernelWorkGroupInfo");
		return 1;
	}

	global_size = count;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	if (err < 0) {
		perror("clEnqueueNDRangeKernel");
		return 1;
	}

	clFinish(queue);

	err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(output), output, 0, NULL, NULL);
	if (err < 0) {
		perror("clEnqueueReadBuffer");
		return 1;
	}

	// Check results
	size_t bad = 0;
	for (size_t i = 0; i < count; ++i) {
		if (output[i] != input[i] * input[i])
			++bad;
	}

	if (bad) {
		fprintf(stderr, "Bad: %lu\n", bad);
		return 1;
	}

	printf("OK\n");

	clReleaseKernel(kernel);
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseCommandQueue(queue);
	clReleaseProgram(program);
	clReleaseContext(context);

	return 0;
}
