#pragma once

#include "mat.h"

#include <assert.h>

struct matview_t {
    using ValueType = u64;
    using ValueRef = ValueType&;
    using ValuePtr = ValueType*;

    matview_t() = default;

    matview_t(const mat_t &m)
    :data(m.data.get())
    ,width(m.width)
    ,height(m.height)
    ,stride(m.stride)
    {}

    ValueRef get(u32 x, u32 y) const
    {
        assert(x < this->width);
        assert(y < this->height);

        return this->data[x + y*this->stride];
    }

    ValuePtr data;
    u32 width;
    u32 height;
    u32 stride;
};
