#include "types.h"

#include <random>

namespace {

struct {
    std::mt19937_64 gen;
} context_random;

}

__attribute__((constructor))
static void seed_rand_gen()
{
    context_random.gen.seed(std::random_device()());
}

void memset_random(void *out_, u32 size)
{
    const u32 size_u64 = size / sizeof(u64);
    const u32 size_rem = size % sizeof(u64);

    auto* out_u64 = static_cast<u64*>(out_);
    auto* const out_u64_end = out_u64 + size_u64;

    auto* out_u8 = reinterpret_cast<u8*>(out_u64 + size_u64);
    auto* const out_u8_end = out_u8 + size_rem;

    while (out_u64 != out_u64_end) {
        const u64 random_u64 = (static_cast<u64>(rand()) << 32) | static_cast<u64>(rand());
        *out_u64 = random_u64;

        ++out_u64;
    }

    while (out_u8 != out_u8_end) {
        *out_u8 = static_cast<u8>(rand());
        ++out_u8;
    }
}

