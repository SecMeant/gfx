#include <stdio.h>
#include <string.h>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion"

#include <opencv2/opencv.hpp>

#pragma GCC diagnostic pop

#include "timing.h"
#include "types.h"
#include "config.h"

static constexpr u32 IMAGE_WIDTH = 3840;
static constexpr u32 IMAGE_HEIGHT = 2160;
static constexpr u32 IMAGE_BYTES_PER_PIXEL = 4;
static constexpr u32 IMAGE_SIZE_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_BYTES_PER_PIXEL;

enum class render_target_t {
    GPU,
    CPU,
};

struct program_options_t {
    render_target_t render_target = render_target_t::GPU;
    u32 nr_threads = 1;
    u32 debug : 1 = 0;
    u32 render_image : 1 = 1;
} static opts;

static void
parse_args(const i32 argc, char** const argv)
{
    for (i32 arg_idx = 1; arg_idx < argc; ++arg_idx) {
        if (strcmp(argv[arg_idx], "--cpu") == 0) {
            opts.render_target = render_target_t::CPU;
            continue;
        }

        if (strcmp(argv[arg_idx], "--gpu") == 0) {
            opts.render_target = render_target_t::GPU;
            continue;
        }

        if (strcmp(argv[arg_idx], "--threads") == 0) {
            if (arg_idx+1 >= argc || sscanf(argv[arg_idx+1], "%u", &opts.nr_threads) != 1) {
                fprintf(stderr, "--threads require an argument\n");
                exit(1);
            }

            ++arg_idx;
            continue;
        }

        if (strcmp(argv[arg_idx], "--no-image") == 0) {
            opts.render_image = 0;
            continue;
        }

        if (strcmp(argv[arg_idx], "--debug") == 0) {
            opts.debug = 1;
            continue;
        }
    }
}

#include "render.cc"
#include "render_opencl.cc"
#include "render_cpu.cc"

int
main(int argc, char **argv)
{
    /*
     * Parse arguments
     */
    parse_args(argc, argv);

    /*
     * Prepare the buffers
     */
    std::vector<u8> bitmap_data(IMAGE_SIZE_BYTES, 0);

    timing_info_t tinfo;
    tinfo.reserve(8);

    /*
     * Render the image in memory
     */
    auto& total_render_time = tinfo.emplace_back("total");
    const int render_result = [&]{

        switch (opts.render_target) {

        case render_target_t::GPU:
            return bitmap_render_cl(IMAGE_WIDTH, IMAGE_HEIGHT, std::data(bitmap_data), tinfo);

        case render_target_t::CPU:
            return bitmap_render_cpu(IMAGE_WIDTH, IMAGE_HEIGHT, std::data(bitmap_data), opts.nr_threads, tinfo);
        }

        __builtin_unreachable();
    }();

    total_render_time.stop();

    for (const auto& e : tinfo) {
        printf("%s: %luus\n", e.get_name(), e.get_duration_micro());
    }

    if (render_result)
        return render_result;

    /*
     * Draw the rendered bitmap on screen or save to the file
     */
    if (opts.render_image)
        bitmap_save(std::data(bitmap_data), IMAGE_WIDTH, IMAGE_HEIGHT);

    return 0;
}

