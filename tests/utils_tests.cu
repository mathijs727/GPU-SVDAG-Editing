#include "configuration/hash_dag_enum.h"
#include "cuda_error_check.h"
#include "utils.h"
#include <catch2/catch_all.hpp>

TEST_CASE("morton3D", "[Utils]")
{
    REQUIRE(Utils::morton3D(0b11, 0b11, 0b11) == 0b111111);

    REQUIRE(Utils::morton3D(0b00, 0b11, 0b11) == 0b011011);
    REQUIRE(Utils::morton3D(0b11, 0b00, 0b11) == 0b101101);
    REQUIRE(Utils::morton3D(0b11, 0b11, 0b00) == 0b110110);
    REQUIRE(Utils::morton3D(0b100000000000, 0b100000000000, 0b100000000000) == 0);

    REQUIRE(Utils::morton3D_64(0b10000000000, 0, 0) == 0b100000000000000000000000000000000);
    REQUIRE(Utils::morton3D_64(0, 0b10000000000, 0) == 0b010000000000000000000000000000000);
    REQUIRE(Utils::morton3D_64(0, 0, 0b10000000000) == 0b001000000000000000000000000000000);
    REQUIRE(Utils::morton3D_64(0b100000000000, 0, 0) == 0b100000000000000000000000000000000000);
    REQUIRE(Utils::morton3D_64(0, 0b100000000000, 0) == 0b010000000000000000000000000000000000);
    REQUIRE(Utils::morton3D_64(0, 0, 0b100000000000) == 0b001000000000000000000000000000000000);
    REQUIRE(Utils::morton3D_64(0b100000000000, 0b100000000000, 0b100000000000) == 0b111000000000000000000000000000000000);

    for (uint32_t i = 0; i < 21; ++i) {
        REQUIRE(Utils::morton3D_64(1 << i, 0, 0) == (uint64_t)1 << (3 * i + 2));
        REQUIRE(Utils::morton3D_64(0, 1 << i, 0) == (uint64_t)1 << (3 * i + 1));
        REQUIRE(Utils::morton3D_64(0, 0, 1 << i) == (uint64_t)1 << (3 * i + 0));
    }
}

TEST_CASE("lower_bound", "[Utils]")
{
    std::array data { 1, 4, 8, 19, 27, 42, 69, 102 };
    REQUIRE(Utils::lower_bound<int>(data, 0) == 0);
    REQUIRE(Utils::lower_bound<int>(data, 1) == 0);
    REQUIRE(std::distance(std::begin(data), std::lower_bound(std::begin(data), std::end(data), 1)) == 0);
    REQUIRE(Utils::lower_bound<int>(data, 2) == 1);
    REQUIRE(Utils::lower_bound<int>(data, 3) == 1);
    REQUIRE(Utils::lower_bound<int>(data, 4) == 1);
    REQUIRE(Utils::lower_bound<int>(data, 5) == 2);

    REQUIRE(Utils::lower_bound<int>(data, 0) == 0);
    REQUIRE(Utils::lower_bound<int>(data, 102) == 7);
    REQUIRE(Utils::lower_bound<int>(data, 103) == 8);
}

TEST_CASE("upper_bound", "[Utils]")
{
    std::array data { 1, 4, 8, 19, 27, 42, 69, 102 };
    REQUIRE(Utils::upper_bound<int>(data, 0) == 0);
    REQUIRE(Utils::upper_bound<int>(data, 1) == 1);
    REQUIRE(std::distance(std::begin(data), std::upper_bound(std::begin(data), std::end(data), 1)) == 1);
    REQUIRE(Utils::upper_bound<int>(data, 2) == 1);
    REQUIRE(Utils::upper_bound<int>(data, 3) == 1);
    REQUIRE(Utils::upper_bound<int>(data, 4) == 2);
    REQUIRE(Utils::upper_bound<int>(data, 5) == 2);

    REQUIRE(Utils::upper_bound<int>(data, 0) == 0);
    REQUIRE(Utils::upper_bound<int>(data, 101) == 7);
    REQUIRE(Utils::upper_bound<int>(data, 102) == 8);
    REQUIRE(Utils::upper_bound<int>(data, 103) == 8);
}

TEST_CASE("equal", "[Utils]")
{
    std::array data0 { 1, 4, 8, 19, 27, 42, 69, 102 };
    std::array data1 { 1, 4, 8, 19, 82, 42, 69, 102 };
    std::array data2 { 1, 8, 5, 13, 39, 98, 10, 13 };

    REQUIRE(Utils::equal(std::begin(data0), std::end(data0), std::begin(data0)));
    REQUIRE(!Utils::equal(std::begin(data0), std::end(data0), std::begin(data1)));
    REQUIRE(!Utils::equal(std::begin(data0), std::end(data0), std::begin(data2)));

    REQUIRE(!Utils::equal(std::begin(data1), std::end(data1), std::begin(data0)));
    REQUIRE(Utils::equal(std::begin(data1), std::end(data1), std::begin(data1)));
    REQUIRE(!Utils::equal(std::begin(data1), std::end(data1), std::begin(data2)));

    REQUIRE(!Utils::equal(std::begin(data2), std::end(data2), std::begin(data0)));
    REQUIRE(!Utils::equal(std::begin(data2), std::end(data2), std::begin(data1)));
    REQUIRE(Utils::equal(std::begin(data2), std::end(data2), std::begin(data2)));
}

TEST_CASE("rotate_left", "[Utils]")
{
    REQUIRE(Utils::rotate_left(0b1010100101, 0) == 0b1010100101);
    REQUIRE(Utils::rotate_left(0b1010100101, 1) == 0b10101001010);
    REQUIRE(Utils::rotate_left(0b1010100101, 2) == 0b101010010100);

    REQUIRE(Utils::rotate_left(0x01234567, 4) == 0x12345670);
    REQUIRE(Utils::rotate_left(0x01234567, 8) == 0x23456701);
    REQUIRE(Utils::rotate_left(0x01234567, 12) == 0x34567012);
    REQUIRE(Utils::rotate_left(0x01234567, 16) == 0x45670123);
}

TEST_CASE("rotate_right", "[Utils]")
{
    REQUIRE(Utils::rotate_right(0b1010010100, 0) == 0b1010010100);
    REQUIRE(Utils::rotate_right(0b1010010100, 1) == 0b0101001010);
    REQUIRE(Utils::rotate_right(0b1010010100, 2) == 0b0010100101);

    REQUIRE(Utils::rotate_right(0x01234567, 0) == 0x01234567);
    REQUIRE(Utils::rotate_right(0x01234567, 4) == 0x70123456);
    REQUIRE(Utils::rotate_right(0x01234567, 8) == 0x67012345);
    REQUIRE(Utils::rotate_right(0x01234567, 12) == 0x56701234);
    REQUIRE(Utils::rotate_right(0x01234567, 16) == 0x45670123);
}

TEST_CASE("bit_width", "[Utils]")
{
    REQUIRE(Utils::bit_width(0) == 0);
    REQUIRE(Utils::bit_width(0xFFFFFFFF) == 32);
    REQUIRE(Utils::bit_width(0xFFFFF0FF) == 32);
    REQUIRE(Utils::bit_width(0x0FFFFFFF) == 28);
    REQUIRE(Utils::bit_width(0x08FFFFFF) == 28);
    REQUIRE(Utils::bit_width(0x04FFFFFF) == 27);
    REQUIRE(Utils::bit_width(0b1010010100) == 10);
}

TEST_CASE("popc", "[Utils]")
{
    REQUIRE(Utils::popc(0) == 0);
    REQUIRE(Utils::popc(0xFFFFFFFF) == 32);
    REQUIRE(Utils::popc(0xFFFFF0FF) == 28);
    REQUIRE(Utils::popc(0xFF0FF0FF) == 24);
    REQUIRE(Utils::popc(0b1010010100) == 4);
}

TEST_CASE("popcll", "[Utils]")
{
    REQUIRE(Utils::popcll(0) == 0);
    REQUIRE(Utils::popcll(0xFFFFFFFF) == 32);
    REQUIRE(Utils::popcll(0xFFFFFFFFFF) == 40);
    REQUIRE(Utils::popcll(0xFFFFF0FFFF) == 36);
    REQUIRE(Utils::popcll(0xFF0FF0FFFF) == 32);
    REQUIRE(Utils::popcll(0b1010010100) == 4);
}

TEST_CASE("murmurhash32xN<mask>", "[Utils]")
{
    std::array<uint32_t, 3> lhs, rhs;
    rhs[0] = lhs[0] = 0x1234;
    rhs[1] = lhs[1] = 0x5678;
    rhs[2] = lhs[2] = 0x9012;
    REQUIRE(Utils::murmurhash32xN(lhs.data(), lhs.size()) == Utils::murmurhash32xN(rhs.data(), rhs.size()));

    rhs[0] = 0x1204;
    REQUIRE(Utils::murmurhash32xN(lhs.data(), lhs.size()) != Utils::murmurhash32xN(rhs.data(), rhs.size()));

    /*REQUIRE(Utils::murmurhash32xN<0xFF0F>(lhs.data(), lhs.size()) == Utils::murmurhash32xN<0xFF0F>(rhs.data(), rhs.size()));
    REQUIRE(Utils::murmurhash32xN<0xFFFF>(lhs.data(), lhs.size()) != Utils::murmurhash32xN<0xFFFF>(rhs.data(), rhs.size()));
    REQUIRE(Utils::murmurhash32xN<0x0000>(lhs.data(), lhs.size()) == Utils::murmurhash32xN<0x0000>(rhs.data(), rhs.size()));*/

    rhs[0] = 0x1234;
    rhs[1] = 0x1234;
    REQUIRE(Utils::murmurhash32xN(lhs.data(), lhs.size()) != Utils::murmurhash32xN(rhs.data(), rhs.size()));
    /*REQUIRE(Utils::murmurhash32xN<0xFFFFFFFF>(lhs.data(), lhs.size()) != Utils::murmurhash32xN<0xFFFFFFFF>(rhs.data(), rhs.size()));
    REQUIRE(Utils::murmurhash32xN<0xFFFFFFFF, 0xFFFF0000>(lhs.data(), lhs.size()) == Utils::murmurhash32xN<0xFFFFFFFF, 0xFFFF0000>(rhs.data(), rhs.size()));*/
}

static uint32_t computeHash(const uint32_t* pItem, uint32_t itemSize, HashMethod hashMethod)
{
    switch (hashMethod) {
    case HashMethod::Murmur: {
        return Utils::murmurhash32xN(pItem, itemSize);
    } break;
    case HashMethod::SlabHashXor: {
        return Utils::xorHash32xN(pItem, itemSize);
    } break;
    case HashMethod::SlabHashBoostCombine: {
        return Utils::boostCombineHash32xN(pItem, itemSize);
    } break;
    case HashMethod::SlabHashSingle: {
        return Utils::slabHash(pItem[0]);
    } break;
    default: {
        checkAlways(false);
        return 0;
    }
    };
}

static __global__ void computeHashWarp_kernel(const uint32_t* pItem, uint32_t itemSize, uint32_t* pOut, HashMethod hashMethod)
{
    const uint32_t threadRank = threadIdx.x;
    checkAlways(__activemask() == 0xFFFFFFFF);

    uint32_t out = 0;
    switch (hashMethod) {
    case HashMethod::Murmur: {
        out = Utils::murmurhash32xN(pItem, itemSize);
    } break;
    case HashMethod::SlabHashXor: {
        out = Utils::xorHash32xN_warp(pItem[threadRank], itemSize, threadRank);
    } break;
    case HashMethod::SlabHashBoostCombine: {
        out = Utils::boostCombineHash32xN_warp(pItem[threadRank], itemSize, threadRank);
    } break;
    case HashMethod::SlabHashSingle: {
        out = Utils::slabHash(pItem[0]);
    } break;
    };
    *pOut = out;
}

TEST_CASE("xorHash32xN matches xorHash32xN_warp", "[Utils]")
{
    const auto item = GENERATE(take(50, chunk(32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));
    // Would be nice not to have to hard-code this using magic_enum, but that library doesn't work with the NVCC compiler.
    const std::array hashMethods {
        HashMethod::Murmur,
        HashMethod::SlabHashXor,
        HashMethod::SlabHashBoostCombine,
        HashMethod::SlabHashSingle
    };
    for (uint32_t itemSizeInU32 = 1; itemSizeInU32 < 32; ++itemSizeInU32) {
        for (HashMethod hashMethod : hashMethods) {
            const auto cpuHash = computeHash(item.data(), itemSizeInU32, hashMethod);

            uint32_t *pItemGPU, *pOutGPU;
            cudaMalloc(&pItemGPU, 32 * sizeof(uint32_t));
            cudaMalloc(&pOutGPU, sizeof(uint32_t));
            cudaMemcpy(pItemGPU, item.data(), itemSizeInU32 * sizeof(uint32_t), cudaMemcpyHostToDevice);
            computeHashWarp_kernel<<<1, 32>>>(pItemGPU, itemSizeInU32, pOutGPU, hashMethod);
            CUDA_CHECK_ERROR();
            uint32_t gpuHash;
            cudaMemcpy(&gpuHash, pOutGPU, sizeof(gpuHash), cudaMemcpyDeviceToHost);
            cudaFree(pItemGPU);
            cudaFree(pOutGPU);

            CAPTURE(itemSizeInU32, item, hashMethod);
            CHECK(cpuHash == gpuHash);
        }
    }
}

TEST_CASE("compare_u32_array<mask>", "[Utils]")
{
    std::array<uint32_t, 3> lhs, rhs;
    rhs[0] = lhs[0] = 0x1234;
    rhs[1] = lhs[1] = 0x5678;
    rhs[2] = lhs[2] = 0x9012;
    REQUIRE(Utils::compare_u32_array(lhs.data(), rhs.data(), (uint32_t)lhs.size()));

    rhs[0] = 0x1204;
    REQUIRE(!Utils::compare_u32_array(lhs.data(), rhs.data(), (uint32_t)lhs.size()));

    /*REQUIRE(Utils::compare_u32_array<0xFF0F>(lhs.data(), rhs.data(), (uint32_t)lhs.size()));
    REQUIRE(!Utils::compare_u32_array<0xFFFF>(lhs.data(), rhs.data(), (uint32_t)lhs.size()));
    REQUIRE(Utils::compare_u32_array<0x0000>(lhs.data(), rhs.data(), (uint32_t)lhs.size()));*/

    rhs[0] = 0x1234;
    rhs[1] = 0x1234;
    REQUIRE(!Utils::compare_u32_array(lhs.data(), rhs.data(), (uint32_t)lhs.size()));
    /*REQUIRE(!Utils::compare_u32_array<0xFFFFFFFF>(lhs.data(), rhs.data(), (uint32_t)lhs.size()));
    REQUIRE(Utils::compare_u32_array<0xFFFFFFFF, 0xFFFF0000>(lhs.data(), rhs.data(), (uint32_t)lhs.size()));*/
}
