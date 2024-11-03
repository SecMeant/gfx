#pragma once

#include "random.h"
#include "types.h"

#include <assert.h>
#include <memory>
#include <algorithm>
#include <vector>

struct mat_i64_t {
    using ValueType = i64;
    using ValueRef = ValueType&;
    using ValueCRef = const ValueType&;
    using ValueCPtr = const ValueType*;
    using InitializerType = std::vector<std::vector<ValueType>>;

    mat_i64_t() = default;

    mat_i64_t(mat_i64_t &&other)
    :data(std::move(other.data))
    ,width(other.width)
    ,height(other.height)
    ,stride(other.stride)
    { }

    mat_i64_t& operator=(mat_i64_t &&other)
    {
        this->data = std::move(other.data);
        this->width = other.width;
        this->height = other.height;
        this->stride = other.stride;

        return *this;
    }

    mat_i64_t(const InitializerType& init)
    {
        const auto height = init.size();
        const auto width = height == 0 ? 0 : init[0].size();

        *this = make_matrix(width, height);

        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x)
                this->at(x,y) = init[y][x];
    }

    static mat_i64_t make_matrix(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        assert(stride >= width);

        mat_i64_t ret;

        ret.data = std::make_unique<ValueType[]>(stride * height);
        ret.width = width;
        ret.height = height;
        ret.stride = stride;

        return ret;
    }

    static mat_i64_t make_matrix_from_data(const i32 *data, const u32 width, const u32 height, u32 stride = 0)
    {
        /* TODO: We don't always have to reallocate */
        mat_i64_t ret = make_matrix(width, height, stride);

        auto src_row = data;
        auto dst_row = ret.data.get();
        for (u32 y = 0; y < height; ++y) {
            std::copy_n(src_row, width, dst_row);
            src_row += width;
            dst_row += ret.stride;
        }

        return ret;
    }

    static u32 gen_stride(u32 width)
    {
        return (width + 15UL) & (~15UL);
    }

    static mat_i64_t make_matrix_zero(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        mat_i64_t ret = make_matrix(width, height, stride);

        ret.set_zero();

        return ret;
    }

    static mat_i64_t make_matrix_random(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        mat_i64_t ret = make_matrix(width, height, stride);

        ret.set_random();

        return ret;
    }

    size_t num_elems() const
    {
        return this->height * this->stride;
    }

    size_t size_bytes() const
    {
        return this->num_elems() * sizeof(ValueType);
    }

    void set_zero()
    {
        std::fill(this->data.get(), this->data.get() + this->num_elems(), 0);
    }

    void set_random()
    {
        memset_random(this->data.get(), this->num_elems() * sizeof(ValueType));
    }

    ValueRef at(u32 x, u32 y)
    {
        return this->data[y * this->stride + x];
    }

    ValueCRef at(u32 x, u32 y) const
    {
        return this->data[y * this->stride + x];
    }

#if __cplusplus >= 202300L
    ValueRef operator[](u32 x, u32 y)
    {
        return this->at(x, y);
    }

    ValueCRef operator[](u32 x, u32 y) const
    {
        return this->at(x, y);
    }
#endif

    std::unique_ptr<ValueType[]> data;
    u32 width;
    u32 height;
    u32 stride;
};

struct matview_i64_t {
    using ValueType = mat_i64_t::ValueType;
    using ValueRef = ValueType&;
    using ValueCRef = const ValueType&;
    using ValuePtr = ValueType*;

    constexpr matview_i64_t()
    : data(nullptr)
    , width(0)
    , height(0)
    , stride(0)
    {}

    matview_i64_t(const matview_i64_t &other) = default;
    matview_i64_t(matview_i64_t &&other) = default;
    matview_i64_t& operator=(const matview_i64_t &other) = default;
    matview_i64_t& operator=(matview_i64_t &&other) = default;

    matview_i64_t(const mat_i64_t &m)
    :data(m.data.get())
    ,width(m.width)
    ,height(m.height)
    ,stride(m.stride)
    {}

    matview_i64_t(ValuePtr data, u32 width, u32 height, u32 stride)
    :data(data)
    ,width(width)
    ,height(height)
    ,stride(stride)
    {}

    ValueRef at(u32 x, u32 y)
    {
        return this->data[y * this->stride + x];
    }

    ValueCRef at(u32 x, u32 y) const
    {
        return this->data[y * this->stride + x];
    }

#if __cplusplus >= 202300L
    ValueRef operator[](u32 x, u32 y)
    {
        return this->at(x, y);
    }

    ValueCRef operator[](u32 x, u32 y) const
    {
        return this->at(x, y);
    }
#endif

    size_t num_elems() const
    {
        return this->height * this->stride;
    }

    size_t size_bytes() const
    {
        return this->num_elems() * sizeof(ValueType);
    }

    ValuePtr data;
    u32 width;
    u32 height;
    u32 stride;
};

constexpr bool mat_dim_match(matview_i64_t m0, matview_i64_t m1)
{
    return m0.width == m1.width && m0.height == m1.height;
}

mat_i64_t mat_add_cpu(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_sub_cpu(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cpu(matview_i64_t lhs, matview_i64_t rhs);
void mat_copy(matview_i64_t dst, matview_i64_t src);

mat_i64_t strassen_cpu(matview_i64_t lhs, matview_i64_t rhs);

mat_i64_t mat_add_cl(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_sub_cl(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cl(matview_i64_t lhs, matview_i64_t rhs);

mat_i64_t mat_mul_cu(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cu_umem_tiled(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cu_tiled(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cu_test(matview_i64_t lhs, matview_i64_t rhs);

