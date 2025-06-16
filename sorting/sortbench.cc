#include <span>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <source_location>
#include <type_traits>
#include <ranges>
#include <vector>
#include <cassert>

#include "types.h"

#define PANIC_ON_FALSE(expr, loc) \
    do { \
        panic_on_false(expr, #expr, loc); \
    } while(0);

struct radix_context {
    using count_type = u32;
    static constexpr size_t RADIX = 16;

    std::array<count_type, RADIX> hist; /* Histogram buffer */
    std::array<count_type, RADIX> pref; /* Prefix buffer */
};

thread_local static struct radix_context rctx;

inline void panic_on_false(
    bool condition,
    const char* expr_str,
    const std::source_location loc = std::source_location::current()
) {
    if (!condition) {
        printf("Assertion failed: %s\n", expr_str);
        printf("  at: %s, %s:%u\n", loc.function_name(), loc.file_name(), loc.line());
        std::exit(1);
    }
}

inline void print_array_mismatch(
    std::span<const u64> arr,
    const size_t center,
    const char* const label,
    const size_t window = 4
) {
    const size_t start = (center < window) ? 0 : center - window;
    const size_t end = std::min(center + window + 1, arr.size());

    printf("  %s [%zu..%zu): ", label, start, end);
    for (size_t i = start; i < end; ++i) {
        if (i == center)
            printf(">>%lu<< ", scast<u64>(arr[i]));
        else
            printf("%lu ", scast<u64>(arr[i]));
    }
    putchar('\n');
}

inline void assert_arrays_equal(
    std::span<const u64> actual,
    std::span<const u64> expected,
    const std::source_location loc = std::source_location::current()
) {
    PANIC_ON_FALSE(actual.size() == expected.size(), loc);

    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            printf("Array content mismatch at index %zu: %lu != %lu\n", i, actual[i], expected[i]);
            print_array_mismatch(actual, i, "Actual   ");
            print_array_mismatch(expected, i, "Expected ");
            std::exit(1);
        }
    }
}

constexpr size_t dec_digit_count(unsigned long long val)
{
    // Powers of 10 thresholds for digit counts 1 to 20 (max for 64-bit)
    constexpr unsigned long long powers_of_10[] = {
        1ULL,                  // 1 digit
        10ULL,
        100ULL,
        1'000ULL,
        10'000ULL,
        100'000ULL,
        1'000'000ULL,
        10'000'000ULL,
        100'000'000ULL,
        1'000'000'000ULL,
        10'000'000'000ULL,
        100'000'000'000ULL,
        1'000'000'000'000ULL,
        10'000'000'000'000ULL,
        100'000'000'000'000ULL,
        1'000'000'000'000'000ULL,
        10'000'000'000'000'000ULL,
        100'000'000'000'000'000ULL,
        1'000'000'000'000'000'000ULL,
        10'000'000'000'000'000'000ULL // overflow but safe
    };

    // Find smallest i with val < powers_of_10[i]
    for (size_t i = 0; i < sizeof(powers_of_10)/sizeof(powers_of_10[0]); ++i) {
        if (val < powers_of_10[i]) {
            return i;
        }
    }

    return 20; // max digits for 64-bit unsigned
}

constexpr size_t hex_digit_count(unsigned long long val)
{
    size_t digits = 1;
    while (val >>= 4) digits++;
    return digits;
}

template <std::ranges::range R>
requires std::is_unsigned_v<std::ranges::range_value_t<R>>
void print_range(const R& data, size_t max_columns = 16, bool print_hex = false)
{
    using T = std::ranges::range_value_t<R>;

    const size_t size = std::ranges::size(data);

    const char * const format = [&]() {
        if (print_hex)
            return "%0*llX ";
        return "%*llu ";
    }();

    size_t max_width = 1;

    for (T val : data) {
        size_t digits;

        if (print_hex) {
            digits = hex_digit_count(scast<u64>(val));
        } else {
            digits = dec_digit_count(scast<u64>(val));
        }

        max_width = std::max(max_width, digits);
    }

    size_t index = 0;
    for (T val : data) {
        if (index % max_columns == 0) {
            printf("  [%04zu]: ", index);
        }

        printf(format, scast<int>(max_width), scast<u64>(val));

        ++index;
        if (index % max_columns == 0 || index == size) {
            printf("\n");
        }
    }
}

static std::vector<u64> clone(std::span<const u64> data)
{
    std::vector<u64> ret(std::begin(data), std::end(data));
    return ret;
}

static constexpr u64 make_nbits_mask(const u32 nbits)
{
    if (nbits == 0)
        return 0;

    return (1u << nbits) - 1u;
}

constexpr inline u32 BITS_PER_BYTE = 8;

void radix_sort_(std::span<u64> data, u32 bitposition)
{
    constexpr u32 num_buckets = 4u;
    const u64 bitmask = make_nbits_mask(2) << bitposition;
    auto data_out = clone(data);

    std::fill(std::begin(rctx.hist), std::end(rctx.hist), 0);

    /* Count */
    for (u32 i = 0; i < std::size(data); ++i) {
        const u32 bucket = (data[i] & bitmask) >> bitposition;
        ++rctx.hist[bucket];
    }

    /* Scan */
    rctx.pref[0] = 0;
    for (u32 i = 1; i < num_buckets; ++i) {
        rctx.pref[i] = rctx.pref[i-1] + rctx.hist[i-1];
    }

    /* Order */
    for (u32 bucket = 0; bucket < num_buckets; ++bucket) {

        const auto num_emit = rctx.hist[bucket];
        const auto offset   = rctx.pref[bucket];
        u32 emitted = 0;

        /* Maintain the stable sort. Emit values in the order of original appearance */
        assert(std::size(data_out) == std::size(data));
        for (u32 j = 0; j < std::size(data) && emitted < num_emit; ++j) {
            if (((data[j] & bitmask) >> bitposition) != bucket)
                continue;

            data_out[emitted + offset] = data[j];
            ++emitted;
        }
    }
    std::copy(std::begin(data_out), std::end(data_out), std::begin(data));
}

void radix_sort(std::span<u64> data)
{
    const u32 bits = sizeof(decltype(data)::element_type) * BITS_PER_BYTE;

    for (u32 bitpos = 0; bitpos < bits; bitpos += 2) {
        print_range(data, 16, false);
        radix_sort_(data, bitpos);
    }
    print_range(data, 16, false);
}

int main()
{
    std::array<u64, 16> data = {
        0x6b00000, 4, 31, 13, 23, 0x7b00000, 22, 48, 3, 1, 2392, 1323204, 8231, 82, 7, 77
    };

    auto expected = data;

    std::sort(std::begin(expected), std::end(expected));
    radix_sort(data);

    assert_arrays_equal(data, expected);
    print_range(data, 16, false);

    return 0;
}

