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

void test_matrix_vs_pytorch(const char * const filepath)
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


    /*
     * Convert safetensor file data to internal format.
     * Compute result and compare against the result from file.
     */
    for (const auto &ttrip: ttrips) {
        const auto& test_id = ttrip.first;
        const auto& tensa   = ttrip.second.a;
        const auto& tensb   = ttrip.second.b;
        const auto& tensc   = ttrip.second.c;

        if (tensa.data == nullptr || tensb.data == nullptr || tensc.data == nullptr)
            throw test_failure(fmt::format("Incomplete data for id{}\n", test_id));


        /* Prepare data from pytorch */
        mat_t mata = make_mat_from_tensor_data(tensa);
        mat_t matb = make_mat_from_tensor_data(tensb);
        mat_t matc_expected = make_mat_from_tensor_data(tensc);


        /* Test using mat_mul_cpu() */
        mat_t matc_computed = mat_mul_cpu(mata, matb);

        TEST_ASSERT(matc_expected.width == matc_computed.width);
        TEST_ASSERT(matc_expected.height == matc_computed.height);

        auto test_name = fmt::format("{}.{}.{}", filepath, test_id, "mat_mul_cpu");
        mat_compare_or_fail(test_name.c_str(), matc_computed, matc_expected, mata, matb, mat_op::mul);


        /* Test using strassen_cpu() */
        mat_t matc_computed_strassen = strassen_cpu(mata, matb);

        TEST_ASSERT(matc_expected.width == matc_computed_strassen.width);
        TEST_ASSERT(matc_expected.height == matc_computed_strassen.height);

        test_name = fmt::format("{}.{}.{}", filepath, test_id, "strassen_cpu");
        mat_compare_or_fail(test_name.c_str(), matc_computed_strassen, matc_expected, mata, matb, mat_op::mul);


        /* Test using opencl kernel */
        mat_t matc_computed_cl = mat_mul_cl(mata, matb);

        TEST_ASSERT(matc_expected.width == matc_computed_cl.width);
        TEST_ASSERT(matc_expected.height == matc_computed_cl.height);

        test_name = fmt::format("{}.{}.{}", filepath, test_id, "cl");
        mat_compare_or_fail(test_name.c_str(), matc_computed_cl, matc_expected, mata, matb, mat_op::mul);


        /* Test using cuda kernel */
        mat_t matc_computed_cu = mat_mul_cu(mata, matb);

        TEST_ASSERT(matc_expected.width == matc_computed_cl.width);
        TEST_ASSERT(matc_expected.height == matc_computed_cl.height);

        test_name = fmt::format("{}.{}.{}", filepath, test_id, "cu");
        mat_compare_or_fail(test_name.c_str(), matc_computed_cu, matc_expected, mata, matb, mat_op::mul);
    }
}

