#include <catch2/catch_all.hpp>
// Include catch2 first.
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"

TEST_CASE("MyGPUHashDAG node header encoding", "[MyGPUHashDAG]")
{
    using DAG = MyGPUHashDAG<EMemoryType::CPU>;

    const uint32_t userPayload = GENERATE(1, 2, 3, 4, 79, 90, 127, 255);
    const uint32_t bitMask = GENERATE(1, 2, 3, 4, 79, 90, 127, 255);

    uint32_t nodeHeader[2];
    uint32_t* pRest = DAG::encode_node_header_with_payload(nodeHeader, bitMask, userPayload);
    REQUIRE(pRest == nodeHeader + 1);

    const uint32_t controlBitMask = DAG::get_node_child_mask(nodeHeader[0]);
    const uint32_t controlUserPayload = DAG::get_node_user_payload(nodeHeader[0]);
    REQUIRE(controlBitMask == bitMask);
    REQUIRE(Utils::child_mask(nodeHeader[0]) == controlBitMask);
    REQUIRE(controlUserPayload == userPayload);
}

#if EDITS_ENABLE_MATERIALS
TEST_CASE("MyGPUHashDAG::LeafBuilder", "[MyGPUHashDAG]")
{
    using DAG = MyGPUHashDAG<EMemoryType::CPU>;
    using LeafBuilder = typename DAG::LeafBuilder;

    SECTION("Fixed value")
    {
        const uint32_t fillMaterial = GENERATE(range(0u, DAG::NumMaterials - 1u));
        const uint64_t materialMask = GENERATE(0x1llu, 0xFFFFFFFFFFFFFFFFllu, 0xAAAAAAAAAAAAAAAAllu, 0xABABABABABABABABllu);

        uint32_t buffer[DAG::maxLeafSizeInU32 + 1];
        std::fill(std::begin(buffer), std::end(buffer), 0);
        LeafBuilder leafBuilder { buffer };
        for (uint32_t i = 0; i < 64; ++i) {
            if (materialMask & (1llu << i))
                leafBuilder.set(fillMaterial);
            leafBuilder.next();
        }
        const uint32_t bufferSize = leafBuilder.finalize();
        REQUIRE(bufferSize >= DAG::minLeafSizeInU32);
        REQUIRE(bufferSize <= DAG::maxLeafSizeInU32);
        check(buffer[bufferSize] == 0);

        const uint32_t expectedBufferSize = DAG::get_leaf_size(buffer);
        REQUIRE(expectedBufferSize == bufferSize);
        for (uint32_t i = expectedBufferSize; i < DAG::maxLeafSizeInU32; ++i)
            buffer[i] = 0xFFFFFFFF;

        for (uint32_t i = 0; i < 64; ++i) {
            uint32_t material = 0;
            bool filled = DAG::get_material(buffer, i, material);
            if (materialMask & (1llu << i)) {
                REQUIRE(filled);
                REQUIRE(material == fillMaterial);
            } else {
                REQUIRE(!filled);
            }
        }
    }

    SECTION("Varying value")
    {
        uint32_t buffer[DAG::maxLeafSizeInU32];
        std::fill(std::begin(buffer), std::end(buffer), 0);
        LeafBuilder leafBuilder { buffer };
        for (uint32_t i = 0; i < 64; ++i) {
            if (i % 3)
                leafBuilder.set(i % 11);
            leafBuilder.next();
        }
        const uint32_t bufferSize = leafBuilder.finalize();
        REQUIRE(bufferSize >= DAG::minLeafSizeInU32);
        REQUIRE(bufferSize <= DAG::maxLeafSizeInU32);
        check(buffer[bufferSize] == 0);

        const uint32_t expectedBufferSize = DAG::get_leaf_size(buffer);
        REQUIRE(expectedBufferSize == bufferSize);
        for (uint32_t i = expectedBufferSize; i < DAG::maxLeafSizeInU32; ++i)
            buffer[i] = 0xFFFFFFFF;

        for (uint32_t i = 0; i < 64; ++i) {
            uint32_t material = 0;
            bool filled = DAG::get_material(buffer, i, material);
            if (i % 3) {
                REQUIRE(filled);
                REQUIRE(material == i % 11);
            } else {
                REQUIRE(!filled);
            }
        }
    }
}

#endif