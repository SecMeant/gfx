#pragma once

#include "random.h"
#include "types.h"

#include <assert.h>
#include <memory>

struct mat_t {
    static mat_t make_matrix(const u32 width, const u32 height, const u32 stride)
    {
        mat_t ret;

        assert(stride >= width);

        ret.data = std::make_unique<u64[]>(stride * height);
        ret.width = width;
        ret.height = height;
        ret.stride = stride;

        return ret;
    }

    static mat_t make_matrix_random(const u32 width, const u32 height, const u32 stride)
    {
        mat_t ret = make_matrix(width, height, stride);

        ret.set_random();

        return ret;
    }

    void set_random()
    {
        memset_random(this->data.get(), this->height * this->stride);
    }

    std::unique_ptr<u64[]> data;
    u32 width;
    u32 height;
    u32 stride;
};

