#pragma once

#include "random.h"
#include "types.h"

#include <assert.h>
#include <memory>
#include <algorithm>
#include <vector>

struct mat_t {
    using ValueType = i64;
    using ValueRef = ValueType&;
    using ValueCRef = const ValueType&;
    using InitializerType = std::vector<std::vector<ValueType>>;

    mat_t() = default;

    mat_t(const InitializerType& init)
    {
        const auto height = init.size();
        const auto width = height == 0 ? 0 : init[0].size();

        *this = make_matrix(width, height);

        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x)
                this->operator[](x,y) = init[y][x];
    }

    static mat_t make_matrix(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        assert(stride >= width);

        mat_t ret;

        ret.data = std::make_unique<ValueType[]>(stride * height);
        ret.width = width;
        ret.height = height;
        ret.stride = stride;

        return ret;
    }

    static u32 gen_stride(u32 width)
    {
        return (width + 15UL) & (~15UL);
    }

    static mat_t make_matrix_zero(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        mat_t ret = make_matrix(width, height, stride);

        ret.set_zero();

        return ret;
    }

    static mat_t make_matrix_random(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        mat_t ret = make_matrix(width, height, stride);

        ret.set_random();

        return ret;
    }

    size_t num_elems() const
    {
        return this->height * this->stride;
    }

    void set_zero()
    {
        std::fill(this->data.get(), this->data.get() + this->num_elems(), 0);
    }

    void set_random()
    {
        memset_random(this->data.get(), this->num_elems() * sizeof(ValueType));
    }

    ValueRef operator[](u32 x, u32 y)
    {
        return this->data[y * this->stride + x];
    }

    ValueCRef operator[](u32 x, u32 y) const
    {
        return this->data[y * this->stride + x];
    }

    std::unique_ptr<ValueType[]> data;
    u32 width;
    u32 height;
    u32 stride;
};

struct matview_t {
    using ValueType = mat_t::ValueType;
    using ValueRef = ValueType&;
    using ValueCRef = const ValueType&;
    using ValuePtr = ValueType*;

    matview_t() = default;
    matview_t(const matview_t &other) = default;
    matview_t(matview_t &&other) = default;
    matview_t& operator=(const matview_t &other) = default;
    matview_t& operator=(matview_t &&other) = default;

    matview_t(const mat_t &m)
    :data(m.data.get())
    ,width(m.width)
    ,height(m.height)
    ,stride(m.stride)
    {}

    matview_t(ValuePtr data, u32 width, u32 height, u32 stride)
    :data(data)
    ,width(width)
    ,height(height)
    ,stride(stride)
    {}

    ValueRef operator[](u32 x, u32 y)
    {
        return this->data[y * this->stride + x];
    }

    ValueCRef operator[](u32 x, u32 y) const
    {
        return this->data[y * this->stride + x];
    }

    ValuePtr data;
    u32 width;
    u32 height;
    u32 stride;
};

mat_t mat_add_cpu(matview_t lhs, matview_t rhs);
mat_t mat_sub_cpu(matview_t lhs, matview_t rhs);
mat_t mat_mul_cpu(matview_t lhs, matview_t rhs);
void mat_copy(matview_t dst, matview_t src);

mat_t strassen_cpu(matview_t lhs, matview_t rhs);

