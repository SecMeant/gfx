#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <string>
#include <map>

#include <fmt/format.h>

#include <mipc/file.h>

#include <rapidjson/document.h>

#include "test.h"
#include "mat.h"
#include "types.h"
#include "print_utils.h"
#include "timing.h"
#include "bench.h"
#include "options.h"

struct safetensor {
    enum class dtype {
        none,
        i32,
        f32,
    };

    // Shape
    u32 rows = 0;
    u32 cols = 0;

    dtype type = dtype::none;

    const void* data = nullptr;
};

constexpr static const char* dtype2str(safetensor::dtype t)
{
    using dtype = safetensor::dtype;

    switch(t) {
    case dtype::none:
        return "none";
    case dtype::i32:
        return "i32";
    case dtype::f32:
        return "f32";
    }

    __builtin_unreachable();

    return "none";
}

struct test_tripplet {
    safetensor a;
    safetensor b;
    safetensor c;
};

static mat_i64_t make_mat_i32_from_tensor_data(const safetensor &tensor)
{
    return mat_i64_t::make_matrix_from_data(reinterpret_cast<const i32*>(tensor.data), tensor.cols, tensor.rows);
}

static mat_f32_t make_mat_f32_from_tensor_data(const safetensor &tensor)
{
    return mat_f32_t::make_matrix_from_data(reinterpret_cast<const f32*>(tensor.data), tensor.cols, tensor.rows);
}

static const char* filename_from_path(std::string_view filepath)
{
    const auto pos = filepath.find_last_of('/');
    if (pos == std::string_view::npos)
        return filepath.begin();

    return &filepath[pos+1];
}

static int parse_safetensors(mipc::finbuf &ftensors, std::map<u64, test_tripplet> &ttrips)
{
    const char *begin = ftensors.begin();

    u64 metadata_size = 0;
    std::memcpy(&metadata_size, begin, sizeof(u64));

    begin += sizeof(u64);

    std::string header_str;
    header_str.assign(begin, metadata_size+1);
    header_str[metadata_size] = '\0';

    rapidjson::Document header_json;
    header_json.Parse(header_str.c_str());

    assert(header_json.IsObject());
    for (auto &m: header_json.GetObject()) {
        char c;
        u64 id;

        if (sscanf(m.name.GetString(), "%c%lu", &c, &id) != 2)
            throw test_failure(fmt::format("Expected %c%lu, got: {}\n", m.name.GetString()));

        auto& ttrip = ttrips[id];

        /*
         * "A":{
         *  "dtype":"I32",
         *  "shape":[128,128],
         *  "data_offsets":[0,65536]
         * },
         */

        auto parse_tensor_data = [&] (safetensor &t) {

            /*
             * Parse dtype
             */
            if (!m.value.HasMember("dtype")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "dtype");
                return 1;
            }

            if (!m.value["dtype"].IsString()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                       m.name.GetString(), "dtype", "string");
                return 1;
            }

            t.type = safetensor::dtype::none;
            const char * const typestr = m.value["dtype"].GetString();

            if (strcmp(typestr, "I32") == 0)
                t.type = safetensor::dtype::i32;

            if (strcmp(typestr, "F32") == 0)
                t.type = safetensor::dtype::f32;

            if (t.type == safetensor::dtype::none) {
                fmt::print(stderr, "{}: Expected {} field to be \"I32\" or \"F32\", but got {}\n",
                       m.name.GetString(), "dtype", m.value.GetString());
                return 1;
            }


            /*
             * Parse shape
             */
            if (!m.value.HasMember("shape")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "shape");
                return 1;
            }

            if (!m.value["shape"].IsArray()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                       m.name.GetString(), "shape", "array");
                return 1;
            }

            if (m.value["shape"].GetArray().Size() != 2) {
                fmt::print(stderr, "{}: Expected shape to be array of size 2\n",
                       m.name.GetString());
                return 1;
            }

            t.rows = m.value["shape"].GetArray()[0].GetUint();
            t.cols = m.value["shape"].GetArray()[1].GetUint();


            /*
             * Parse data
             */
            if (!m.value.HasMember("data_offsets")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "data_offsets");
                return 1;
            }

            if (!m.value["data_offsets"].IsArray()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                       m.name.GetString(), "data_offsets", "array");
                return 1;
            }

            if (m.value["data_offsets"].GetArray().Size() != 2) {
                fmt::print(stderr, "{}: Expected data_offsets to be array of size 2\n",
                       m.name.GetString());
                return 1;
            }

            t.data = rcast<const void*>(
                /* Get data offset as base. */
                m.value["data_offsets"].GetArray()[0].GetUint64() +

                /* Skip metadata size. */
                sizeof(u64) +

                /* Skip the actual metadata. */
                metadata_size +

                /* Finally, offset from base of mmaped file to obtain data pointer. */
                ftensors.begin()
            );

            return 0;
        };

        switch (c) {
        case 'A':
            if (parse_tensor_data(ttrip.a))
                return 1;
            break;

        case 'B':
            if (parse_tensor_data(ttrip.b))
                return 1;
            break;

        case 'C':
            if (parse_tensor_data(ttrip.c))
                return 1;
            break;

        default:
            fmt::print(stderr, "SKIP: Unexpected matrix name: {}\n", m.name.GetString());
            break;
        }
    }

    return 0;
}

void test_matrix_vs_pytorch_i32(const char * const filepath, test_flags_t flags)
{
    auto ftensors = mipc::finbuf(filepath);
    if (!ftensors)
        throw test_failure("Failed to open safetensors file\n");

    std::map<u64, test_tripplet> ttrips;
    if (parse_safetensors(ftensors, ttrips))
        return;

    const char * const filename = filename_from_path(filepath);
    timeit_t timer;

    /* For calculating averages. */
    auto dur_cpu             = timeit_t::Duration::zero();
    auto dur_strassen_cpu    = timeit_t::Duration::zero();
    auto dur_cl              = timeit_t::Duration::zero();
    auto dur_cuda            = timeit_t::Duration::zero();
    auto dur_cuda_umem_tiled = timeit_t::Duration::zero();
    auto dur_cuda_tiled      = timeit_t::Duration::zero();
    auto dur_cuda_tiled_in   = timeit_t::Duration::zero();
    auto dur_cuda_test       = timeit_t::Duration::zero();


    /*
     * Convert safetensor file data to internal format.
     * Compute result and compare against the result from file.
     */
    for (const auto &ttrip: ttrips) {
        const auto& test_id = ttrip.first;
        const auto& tensa   = ttrip.second.a;
        const auto& tensb   = ttrip.second.b;
        const auto& tensc   = ttrip.second.c;

        std::string test_name;
        const bool run_on_cpu = !flags.skip_cpu;
        const bool run_opencl = true;
        const bool run_cuda = true;

        if (tensa.data == nullptr || tensb.data == nullptr || tensc.data == nullptr)
            throw test_failure(fmt::format("Incomplete data for id{}\n", test_id));

        using dtype = safetensor::dtype;
        if (tensa.type != dtype::i32 || tensb.type != dtype::i32 || tensc.type != dtype::i32)
            throw test_failure(fmt::format(
                "Mismatched tensor types for id{}. "
                "Expected all to be i32, got A.dtype = {}, B.dtype = {}, C.dtype = {}",
                test_id, dtype2str(tensa.type), dtype2str(tensb.type), dtype2str(tensc.type)
            ));


        /* Prepare data from pytorch */
        mat_i64_t mata = make_mat_i32_from_tensor_data(tensa);
        mat_i64_t matb = make_mat_i32_from_tensor_data(tensb);
        mat_i64_t matc_expected = make_mat_i32_from_tensor_data(tensc);


        if (run_on_cpu) {
            /* Test using mat_mul_cpu() */
            timer.start();
            mat_i64_t matc_computed = mat_mul_cpu(mata, matb);
            timer.stop();

            dur_cpu += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed.width);
            TEST_ASSERT(matc_expected.height == matc_computed.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "mat_mul_cpu");
            mat_compare_or_fail(test_name.c_str(), matc_computed, matc_expected, mata, matb, mat_op::mul);


            /* Test using strassen_cpu() */
            timer.start();
            mat_i64_t matc_computed_strassen = strassen_cpu(mata, matb);
            timer.stop();

            dur_strassen_cpu += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_strassen.width);
            TEST_ASSERT(matc_expected.height == matc_computed_strassen.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "strassen_cpu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_strassen, matc_expected, mata, matb, mat_op::mul);
        }


        if (run_opencl) {
            /* Test using opencl kernel */
            timer.start();
            mat_i64_t matc_computed_cl = mat_mul_cl(mata, matb);
            timer.stop();

            dur_cl += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cl.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cl.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cl");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cl, matc_expected, mata, matb, mat_op::mul);
        }


        if (run_cuda) {
            /* Test using cuda kernel */
            timer.start();
            mat_i64_t matc_computed_cu = mat_mul_cu(mata, matb);
            timer.stop();

            dur_cuda += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);


            /* Test using cuda kernel (tiled, uniform memory) */
            timer.start();
            matc_computed_cu = mat_mul_cu_umem_tiled(mata, matb);
            timer.stop();

            dur_cuda_umem_tiled += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);


            /* Test using cuda kernel (tiled, cache_writes, device memory) */
            timer.start();
            matc_computed_cu = mat_mul_cu_tiled(mata, matb);
            timer.stop();

            dur_cuda_tiled += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);


            /* Test using cuda kernel (tiled, don't cache writes, device memory) */
            timer.start();
            matc_computed_cu = mat_mul_cu_tiled_input(mata, matb);
            timer.stop();

            dur_cuda_tiled_in += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);


            if (opt_test) {
                /* Test using cuda kernel (test) */
                timer.start();
                matc_computed_cu = mat_mul_cu_test(mata, matb);
                timer.stop();

                dur_cuda_test += timer.get_duration();

                TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
                TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

                test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
                mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);
            }
        }
    }


    /*
     * Compute averages
     */

    const auto num_runs = ttrips.size();
    constexpr u32 align = 36u;

    if (dur_cpu.count())
        benchinfo.add(fmt::format("{: <{}}cpu", filename, align), dur_cpu / num_runs);

    if (dur_strassen_cpu.count())
        benchinfo.add(fmt::format("{: <{}}strassen_cpu", filename, align), dur_strassen_cpu / num_runs);

    if (dur_cl.count())
        benchinfo.add(fmt::format("{: <{}}opencl", filename, align), dur_cl / num_runs);

    if (dur_cuda.count())
        benchinfo.add(fmt::format("{: <{}}cuda", filename, align), dur_cuda / num_runs);

    if (dur_cuda_umem_tiled.count())
        benchinfo.add(fmt::format("{: <{}}cuda_umem_tiled_25k", filename, align), dur_cuda_umem_tiled / num_runs);

    if (dur_cuda_tiled.count())
        benchinfo.add(fmt::format("{: <{}}cuda_tiled_25k", filename, align), dur_cuda_tiled / num_runs);

    if (dur_cuda_tiled_in.count())
        benchinfo.add(fmt::format("{: <{}}cuda_tiled_in_25k", filename, align), dur_cuda_tiled_in / num_runs);

    if (dur_cuda_test.count())
        benchinfo.add(fmt::format("{: <{}}cuda_test_25k", filename, align), dur_cuda_test / num_runs);
}

void test_matrix_vs_pytorch_f32(const char * const filepath, test_flags_t flags)
{
    auto ftensors = mipc::finbuf(filepath);
    if (!ftensors)
        throw test_failure("Failed to open safetensors file\n");

    std::map<u64, test_tripplet> ttrips;
    if (parse_safetensors(ftensors, ttrips))
        return;

    const char * const filename = filename_from_path(filepath);
    timeit_t timer;

    /* For calculating averages. */
    auto dur_cpu             = timeit_t::Duration::zero();
    auto dur_strassen_cpu    = timeit_t::Duration::zero();
    auto dur_cl              = timeit_t::Duration::zero();
    auto dur_cuda            = timeit_t::Duration::zero();
    auto dur_cuda_umem_tiled = timeit_t::Duration::zero();
    auto dur_cuda_tiled      = timeit_t::Duration::zero();
    auto dur_cuda_tiled_in   = timeit_t::Duration::zero();
    auto dur_cuda_test       = timeit_t::Duration::zero();

    /*
     * Convert safetensor file data to internal format.
     * Compute result and compare against the result from file.
     */
    for (const auto &ttrip: ttrips) {
        const auto& test_id = ttrip.first;
        const auto& tensa   = ttrip.second.a;
        const auto& tensb   = ttrip.second.b;
        const auto& tensc   = ttrip.second.c;

        std::string test_name;
        const bool run_on_cpu = !flags.skip_cpu;
        const bool run_opencl = true;
        const bool run_cuda = true;

        if (tensa.data == nullptr || tensb.data == nullptr || tensc.data == nullptr)
            throw test_failure(fmt::format("Incomplete data for id{}\n", test_id));

        using dtype = safetensor::dtype;
        if (tensa.type != dtype::f32 || tensb.type != dtype::f32 || tensc.type != dtype::f32)
            throw test_failure(fmt::format(
                "Mismatched tensor types for id{}. "
                "Expected all to be f32, got A.dtype = {}, B.dtype = {}, C.dtype = {}",
                test_id, dtype2str(tensa.type), dtype2str(tensb.type), dtype2str(tensc.type)
            ));

        /* Prepare data from pytorch */
        mat_f32_t mata = make_mat_f32_from_tensor_data(tensa);
        mat_f32_t matb = make_mat_f32_from_tensor_data(tensb);
        mat_f32_t matc_expected = make_mat_f32_from_tensor_data(tensc);

        if (run_on_cpu) {
            /* Test using mat_mul_cpu() */
            timer.start();
            mat_f32_t matc_computed = mat_mul_cpu(mata, matb);
            timer.stop();

            dur_cpu += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed.width);
            TEST_ASSERT(matc_expected.height == matc_computed.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "mat_mul_cpu");
            mat_compare_or_fail(test_name.c_str(), matc_computed, matc_expected, mata, matb, mat_op::mul);


            /* Test using strassen_cpu() */
            timer.start();
            mat_f32_t matc_computed_strassen = strassen_cpu(mata, matb);
            timer.stop();

            dur_strassen_cpu += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_strassen.width);
            TEST_ASSERT(matc_expected.height == matc_computed_strassen.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "strassen_cpu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_strassen, matc_expected, mata, matb, mat_op::mul);
        }

        if (run_opencl) {
            /* Test using opencl kernel */
            timer.start();
            mat_f32_t matc_computed_cl = mat_mul_cl(mata, matb);
            timer.stop();

            dur_cl += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cl.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cl.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cl");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cl, matc_expected, mata, matb, mat_op::mul);
        }

        if (run_cuda) {
            /* Test using cuda kernel */
            timer.start();
            mat_f32_t matc_computed_cu = mat_mul_cu(mata, matb);
            timer.stop();

            dur_cuda += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);


            /* Test using cuda kernel (tiled, uniform memory) */
            timer.start();
            matc_computed_cu = mat_mul_cu_umem_tiled(mata, matb);
            timer.stop();

            dur_cuda_umem_tiled += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);


            /* Test using cuda kernel (tiled, device memory) */
            timer.start();
            matc_computed_cu = mat_mul_cu_tiled(mata, matb);
            timer.stop();

            dur_cuda_tiled += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);


            /* Test using cuda kernel (tiled, device memory) */
            timer.start();
            matc_computed_cu = mat_mul_cu_tiled_input(mata, matb);
            timer.stop();

            dur_cuda_tiled_in += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
            TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
            mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);


            if (opt_test) {
                /* Test using cuda kernel (test) */
                timer.start();
                matc_computed_cu = mat_mul_cu_test(mata, matb);
                timer.stop();

                dur_cuda_test += timer.get_duration();

                TEST_ASSERT(matc_expected.width == matc_computed_cu.width);
                TEST_ASSERT(matc_expected.height == matc_computed_cu.height);

                test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
                mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);
            }
        }
    }


    /*
     * Compute averages
     */

    const auto num_runs = ttrips.size();
    constexpr u32 align = 36u;

    if (dur_cpu.count())
        benchinfo.add(fmt::format("{: <{}}cpu_f32", filename, align), dur_cpu / num_runs);

    if (dur_strassen_cpu.count())
        benchinfo.add(fmt::format("{: <{}}strassen_cpu_f32", filename, align), dur_strassen_cpu / num_runs);

    if (dur_cl.count())
        benchinfo.add(fmt::format("{: <{}}opencl_f32", filename, align), dur_cl / num_runs);

    if (dur_cuda.count())
        benchinfo.add(fmt::format("{: <{}}cuda_f32", filename, align), dur_cuda / num_runs);

    if (dur_cuda_umem_tiled.count())
        benchinfo.add(fmt::format("{: <{}}cuda_umem_tiled_25k_f32", filename, align), dur_cuda_umem_tiled / num_runs);

    if (dur_cuda_tiled.count())
        benchinfo.add(fmt::format("{: <{}}cuda_tiled_25k_f32", filename, align), dur_cuda_tiled / num_runs);

    if (dur_cuda_tiled_in.count())
        benchinfo.add(fmt::format("{: <{}}cuda_tiled_in_25k_f32", filename, align), dur_cuda_tiled_in / num_runs);

    if (dur_cuda_test.count())
        benchinfo.add(fmt::format("{: <{}}cuda_test_25k_f32", filename, align), dur_cuda_test / num_runs);
}
