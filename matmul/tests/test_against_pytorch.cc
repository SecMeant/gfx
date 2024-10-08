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

struct safetensor_i32 {
    // Shape
    u32 rows = 0;
    u32 cols = 0;

    const i32* data = nullptr;
};

struct test_tripplet {
    safetensor_i32 a;
    safetensor_i32 b;
    safetensor_i32 c;
};

static mat_t make_mat_from_tensor_data(const safetensor_i32 &tensor)
{
    return mat_t::make_matrix_from_data(tensor.data, tensor.cols, tensor.rows);
}

static const char* filename_from_path(std::string_view filepath)
{
    const auto pos = filepath.find_last_of('/');
    if (pos == std::string_view::npos)
        return filepath.begin();

    return &filepath[pos+1];
}

void test_matrix_vs_pytorch(const char * const filepath, test_flags_t flags)
{
    auto ftensors = mipc::finbuf(filepath);
    if (!ftensors)
        throw test_failure("Failed to open safetensors file\n");

    const char *begin = ftensors.begin();

    u64 metadata_size = 0;
    std::memcpy(&metadata_size, begin, sizeof(u64));

    begin += sizeof(u64);

    std::string header_str;
    header_str.assign(begin, metadata_size+1);
    header_str[metadata_size] = '\0';

    rapidjson::Document header_json;
    header_json.Parse(header_str.c_str());

    std::map<u64, test_tripplet> ttrips;

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

        auto parse_tensor_data = [&] (safetensor_i32 &t) {

            /*
             * Parse dtype
             */
            if (!m.value.HasMember("dtype")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "dtype");
                return;
            }

            if (!m.value["dtype"].IsString()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                       m.name.GetString(), "dtype", "string");
                return;
            }

            if (strcmp(m.value["dtype"].GetString(), "I32") != 0) {
                fmt::print(stderr, "{}: Expected {} field to be \"I32\", but got {}\n",
                       m.name.GetString(), "dtype", m.value.GetString());
                return;
            }


            /*
             * Parse shape
             */
            if (!m.value.HasMember("shape")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "shape");
                return;
            }

            if (!m.value["shape"].IsArray()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                       m.name.GetString(), "shape", "array");
                return;
            }

            if (m.value["shape"].GetArray().Size() != 2) {
                fmt::print(stderr, "{}: Expected shape to be array of size 2\n",
                       m.name.GetString());
                return;
            }

            t.rows = m.value["shape"].GetArray()[0].GetUint();
            t.cols = m.value["shape"].GetArray()[1].GetUint();


            /*
             * Parse data
             */
            if (!m.value.HasMember("data_offsets")) {
                fmt::print(stderr, "{}: Missing {} field\n", m.name.GetString(), "data_offsets");
                return;
            }

            if (!m.value["data_offsets"].IsArray()) {
                fmt::print(stderr, "{}: Expected {} field to be of type {}\n",
                       m.name.GetString(), "data_offsets", "array");
                return;
            }

            if (m.value["data_offsets"].GetArray().Size() != 2) {
                fmt::print(stderr, "{}: Expected data_offsets to be array of size 2\n",
                       m.name.GetString());
                return;
            }

            t.data = rcast<const i32*>(
                /* Get data offset as base. */
                m.value["data_offsets"].GetArray()[0].GetUint64() +

                /* Skip metadata size. */
                sizeof(u64) +

                /* Skip the actual metadata. */
                metadata_size +

                /* Finally, offset from base of mmaped file to obtain data pointer. */
                ftensors.begin()
            );
        };

        switch (c) {
        case 'A':
            parse_tensor_data(ttrip.a);
            break;

        case 'B':
            parse_tensor_data(ttrip.b);
            break;

        case 'C':
            parse_tensor_data(ttrip.c);
            break;

        default:
            fmt::print(stderr, "SKIP: Unexpected matrix name: {}\n", m.name.GetString());
            break;
        }
    }

    const char * const filename = filename_from_path(filepath);
    timeit_t timer;

    /* For calculating averages. */
    auto dur_cpu             = timeit_t::Duration::zero();
    auto dur_strassen_cpu    = timeit_t::Duration::zero();
    auto dur_cl              = timeit_t::Duration::zero();
    auto dur_cuda            = timeit_t::Duration::zero();
    auto dur_cuda_umem_tiled = timeit_t::Duration::zero();
    auto dur_cuda_tiled      = timeit_t::Duration::zero();
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


        /* Prepare data from pytorch */
        mat_t mata = make_mat_from_tensor_data(tensa);
        mat_t matb = make_mat_from_tensor_data(tensb);
        mat_t matc_expected = make_mat_from_tensor_data(tensc);


        if (run_on_cpu) {
            /* Test using mat_mul_cpu() */
            timer.start();
            mat_t matc_computed = mat_mul_cpu(mata, matb);
            timer.stop();

            dur_cpu += timer.get_duration();

            TEST_ASSERT(matc_expected.width == matc_computed.width);
            TEST_ASSERT(matc_expected.height == matc_computed.height);

            test_name = fmt::format("{}.{}.{}", filepath, test_id, "mat_mul_cpu");
            mat_compare_or_fail(test_name.c_str(), matc_computed, matc_expected, mata, matb, mat_op::mul);


            /* Test using strassen_cpu() */
            timer.start();
            mat_t matc_computed_strassen = strassen_cpu(mata, matb);
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
            mat_t matc_computed_cl = mat_mul_cl(mata, matb);
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
            mat_t matc_computed_cu = mat_mul_cu(mata, matb);
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
    constexpr u32 align = 32u;

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

    if (dur_cuda_test.count())
        benchinfo.add(fmt::format("{: <{}}cuda_test_25k", filename, align), dur_cuda_test / num_runs);
}

