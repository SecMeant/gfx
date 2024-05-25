#include <stdio.h>
#include <math.h>
#include <thread>
#include <vector>
#include <atomic>

#include "timing.h"
#include "types.h"
#include "config.h"

struct thread_render_info_t {
    u32 thread_id;
    u32* bitmap;
    u32 bitmap_offset;
    u32 chunk_size;
    u32 width;
    u32 height;
    std::atomic_flag* start;
};

struct cfloat {
    float x; // real part
    float y; // imaginary

    cfloat operator+(const cfloat &other) const noexcept
    {
        const cfloat &lhs = *this, &rhs = other;

        return cfloat { .x = lhs.x+rhs.x, .y = lhs.y+rhs.y };
    }

    cfloat operator*(const cfloat &other) const noexcept
    {
        const cfloat &lhs = *this, &rhs = other;

        return cfloat { .x = lhs.x*rhs.x - lhs.y*rhs.y, .y = lhs.x*rhs.y + lhs.y*rhs.x };
    }

    float mod() const noexcept
    {
        return std::sqrt(this->x*this->x + this->y*this->y);
    }
};

static cfloat
mandelbrot_step(const cfloat z, const cfloat c)
{
    return z*z + c;
}

static u32
mod2color(const float mod, const float mod_max)
{
    const float scale = (255.0f / mod_max) * mod;
    return 0x010001 * static_cast<u32>(scale);
}

static int
bitmap_render_cpu_on_thread(const thread_render_info_t th_info)
{
    auto *bitmap = th_info.bitmap;
    auto offset = th_info.bitmap_offset;
    const auto offset_end = offset + th_info.chunk_size;
    const auto width = th_info.width;
    const auto height = th_info.height;

    th_info.start->wait(false, std::memory_order_relaxed);

    while (offset != offset_end) {
        cfloat pos = {
            .x = static_cast<float>(offset % width),
            .y = static_cast<float>(offset / width),
        };

        const float zoom = 1.15f;

        /* Scale X */
        pos.x /= width;
        pos.x = pos.x*3.0f - 2.5f;
        pos.x *= zoom;

        /* Scale Y */
        pos.y /= height;
        pos.y = pos.y*2.0f - 1.0f;
        pos.y *= zoom;

        cfloat fout = { .x = 0.0f, .y = 0.0f };
        for (u32 i = 0; i < 32; ++i)
            fout = mandelbrot_step(fout, pos);

        const float cutoff = 0.85f;
        const float fout_mod = std::clamp(fout.mod(), 0.0f, cutoff);

        const u32 color = mod2color(fout_mod, cutoff);
        bitmap[offset] = color;

        ++offset;
    }

    return 0;
}

static int
bitmap_render_cpu(
    const u32 bitmap_width,
    const u32 bitmap_height,
    u8* const bitmap,
    const u32 nr_threads,
    timing_info_t &tinfo
) {
    std::vector<thread_render_info_t> th_infos;
    std::vector<std::thread> threads;
    const auto bitmap_size = bitmap_width * bitmap_height;
    const auto chunk_size_common = bitmap_size / nr_threads;
    const auto chunk_size_last = chunk_size_common + (bitmap_size % nr_threads);
    auto thread_offset = 0;

    /* Thread synchronization */
    std::atomic_flag start = ATOMIC_FLAG_INIT;

    th_infos.reserve(nr_threads);

    if (opts.debug)
        printf("bitmap:\n\twidth:  %u\n\theight: %u\n\tsize:   %u\n\tchunk:  %u\n\trem:    %u\n",
               bitmap_width, bitmap_height, bitmap_size, chunk_size_common, chunk_size_last);

    auto &th_creat_time = tinfo.emplace_back("th_creat");

    for (u32 thread_id = 0; thread_id < nr_threads; ++thread_id) {
        auto &info = th_infos.emplace_back();

        info.thread_id = thread_id;
        info.bitmap = reinterpret_cast<u32*>(bitmap);
        info.bitmap_offset = thread_offset;
        info.chunk_size = thread_id == nr_threads-1 ? chunk_size_last : chunk_size_common;
        info.width = bitmap_width;
        info.height = bitmap_height;

        info.start = &start;

        thread_offset += info.chunk_size;

        if (opts.debug)
            printf("th%u:\n\toffset: %u\n\tchunk:  %u\n",
                   info.thread_id, info.bitmap_offset, info.chunk_size);

        threads.emplace_back(bitmap_render_cpu_on_thread, info);
    }

    th_creat_time.stop();

    auto &th_work_time = tinfo.emplace_back("th_work");

    start.test_and_set();
    start.notify_all();

    for (auto &thread: threads)
        thread.join();

    th_work_time.stop();

    return 0;
}
