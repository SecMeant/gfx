#pragma once

#include "random.h"
#include "types.h"

#include <assert.h>
#include <memory>
#include <algorithm>
#include <vector>

template<typename ValueType_>
struct mat_base_t {
    using ValueType = ValueType_;
    using ValueRef = ValueType&;
    using ValueCRef = const ValueType&;
    using ValueCPtr = const ValueType*;
    using InitializerType = std::vector<std::vector<ValueType>>;

    mat_base_t() = default;

    mat_base_t(mat_base_t &&other)
    :data(std::move(other.data))
    ,width(other.width)
    ,height(other.height)
    ,stride(other.stride)
    { }

    mat_base_t& operator=(mat_base_t &&other)
    {
        this->data = std::move(other.data);
        this->width = other.width;
        this->height = other.height;
        this->stride = other.stride;

        return *this;
    }

    mat_base_t(const InitializerType& init)
    {
        const auto height = init.size();
        const auto width = height == 0 ? 0 : init[0].size();

        *this = make_matrix(width, height);

        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x)
                this->at(x,y) = init[y][x];
    }

    static mat_base_t make_matrix(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        assert(stride >= width);

        mat_base_t ret;

        ret.data = std::make_unique<ValueType[]>(stride * height);
        ret.width = width;
        ret.height = height;
        ret.stride = stride;

        return ret;
    }

    template <typename T>
    static mat_base_t make_matrix_from_data(const T *data, const u32 width, const u32 height, u32 stride = 0)
    {
        /* TODO: We don't always have to reallocate */
        mat_base_t ret = make_matrix(width, height, stride);

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

    static mat_base_t make_matrix_zero(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        mat_base_t ret = make_matrix(width, height, stride);

        ret.set_zero();

        return ret;
    }

    static mat_base_t make_matrix_random(const u32 width, const u32 height, u32 stride = 0)
    {
        if (stride == 0)
            stride = gen_stride(width);

        mat_base_t ret = make_matrix(width, height, stride);

        ret.set_random();

        return ret;
    }

    static bool has_fractional_part(const ValueType v)
    {
        if constexpr (std::is_floating_point_v<ValueType>)
            return false;

        ValueType dummy;
        modf(v, &dummy);

        return v == 0.0f || v == -0.0f;
    }

    static mat_base_t make_matrix_in_range(
        const u32 width,
        const u32 height,
        u32 stride,
        ValueType low,
        ValueType high
    ) {
        if (stride == 0)
            stride = gen_stride(width);

        mat_base_t ret = make_matrix(width, height, stride);

        for (u32 y = 0; y < height; ++y) {
            for (u32 x = 0; x < stride; ++x) {
                ValueType *p = &ret.data.get()[x + y * ret.stride];

                if (x >= width) {
                    *p = 0;
                    continue;
                }

                assert(high > low);
                assert(!has_fractional_part(high));
                assert(!has_fractional_part(low));

                const auto rand_gap = static_cast<int>(high - low);

                if constexpr (std::is_floating_point_v<ValueType>) {
                    const auto rand_0_1 = static_cast<ValueType>(rand() % 1024) / static_cast<ValueType>(1024.0f);
                    *p = (rand_gap * rand_0_1) + low;
                } else {
                    *p = (rand() % rand_gap) + low;
                }
            }
        }

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

using mat_i64_t = mat_base_t<i64>;
using mat_f32_t = mat_base_t<f32>;

template<typename ParentType_>
struct matview_base_t {
    using ParentType = ParentType_;
    using ValueType = typename ParentType::ValueType;
    using ValueRef = ValueType&;
    using ValueCRef = const ValueType&;
    using ValuePtr = ValueType*;

    constexpr matview_base_t()
    : data(nullptr)
    , width(0)
    , height(0)
    , stride(0)
    {}

    matview_base_t(const matview_base_t &other) = default;
    matview_base_t(matview_base_t &&other) = default;
    matview_base_t& operator=(const matview_base_t &other) = default;
    matview_base_t& operator=(matview_base_t &&other) = default;

    constexpr matview_base_t(const ParentType &m)
    :data(m.data.get())
    ,width(m.width)
    ,height(m.height)
    ,stride(m.stride)
    {}

    constexpr matview_base_t(ValuePtr data, u32 width, u32 height, u32 stride)
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

using matview_i64_t = matview_base_t<mat_i64_t>;
using matview_f32_t = matview_base_t<mat_f32_t>;

enum class mat_type_e {
    i64,
    f32,
};

template<> struct matview_base_t<void> {
    constexpr matview_base_t(void *d, u32 w, u32 h, u32 s, mat_type_e t)
    :data(d)
    ,width(w)
    ,height(h)
    ,stride(s)
    ,type(t)
    {}

    matview_base_t(mat_i64_t &m)
    :data(m.data.get())
    ,width(m.width)
    ,height(m.height)
    ,stride(m.stride)
    ,type(mat_type_e::i64)
    {}

    constexpr matview_base_t(matview_i64_t mv)
    :data(mv.data)
    ,width(mv.width)
    ,height(mv.height)
    ,stride(mv.stride)
    ,type(mat_type_e::i64)
    {}

    matview_base_t(mat_f32_t &m)
    :data(m.data.get())
    ,width(m.width)
    ,height(m.height)
    ,stride(m.stride)
    ,type(mat_type_e::f32)
    {}

    constexpr matview_base_t(matview_f32_t mv)
    :data(mv.data)
    ,width(mv.width)
    ,height(mv.height)
    ,stride(mv.stride)
    ,type(mat_type_e::f32)
    {}

    size_t num_elems() const
    {
        return this->height * this->stride;
    }

    size_t size_bytes() const
    {
        switch(this->type) {
        case mat_type_e::i64:
            return this->num_elems() * sizeof(i64);
        case mat_type_e::f32:
            return this->num_elems() * sizeof(f32);
        }

        __builtin_unreachable();
    }

    void *data;
    u32 width;
    u32 height;
    u32 stride;
    mat_type_e type;
};

using matview_void_t = matview_base_t<void>;

template <typename MatViewType>
constexpr bool mat_dim_match(MatViewType m0, MatViewType m1)
{
    return m0.width == m1.width && m0.height == m1.height;
}


/*
 * I32 API
 */

mat_i64_t mat_add_cpu(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_sub_cpu(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cpu(matview_i64_t lhs, matview_i64_t rhs);
void mat_copy(matview_i64_t dst, matview_i64_t src);

mat_i64_t strassen_cpu(matview_i64_t lhs, matview_i64_t rhs);

mat_i64_t mat_mul_cl(matview_i64_t lhs, matview_i64_t rhs);

mat_i64_t mat_mul_cu(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cu_umem_tiled(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cu_tiled(matview_i64_t lhs, matview_i64_t rhs);
mat_i64_t mat_mul_cu_test(matview_i64_t lhs, matview_i64_t rhs);


/*
 * F32 API
 */

mat_f32_t mat_add_cpu(matview_f32_t lhs, matview_f32_t rhs);
mat_f32_t mat_sub_cpu(matview_f32_t lhs, matview_f32_t rhs);
mat_f32_t mat_mul_cpu(matview_f32_t lhs, matview_f32_t rhs);
void mat_copy(matview_f32_t dst, matview_f32_t src);

mat_f32_t strassen_cpu(matview_f32_t lhs, matview_f32_t rhs);

mat_f32_t mat_mul_cl(matview_f32_t lhs, matview_f32_t rhs);

mat_f32_t mat_mul_cu(matview_f32_t lhs, matview_f32_t rhs);
mat_f32_t mat_mul_cu_umem_tiled(matview_f32_t lhs, matview_f32_t rhs);
mat_f32_t mat_mul_cu_tiled(matview_f32_t lhs, matview_f32_t rhs);
mat_f32_t mat_mul_cu_test(matview_f32_t lhs, matview_f32_t rhs);
