#include <catch2/catch_all.hpp>
// Include catch2 first.
#include "array.h"
#include "cuda_helpers.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/duplicate_detection_hash_table.h"
#include "timings.h"
#include <random>

static constexpr uint32_t itemSizeInU32 = 10;

template <typename A, typename B>
static bool compareItem(A lhs, B rhs)
{
    for (uint32_t i = 0; i < itemSizeInU32; ++i) {
        if (lhs[i] != rhs[i])
            return false;
    }
    return true;
}

static __global__ void bulkAddAsWarp_kernel(DuplicateDetectionHashTable table, gsl::span<const uint32_t> items, uint32_t numItems, gsl::span<uint32_t> outLocations)
{
#ifdef __CUDA_ARCH__
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < numItems + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx < numItems && warpIdx * table.itemSizeInU32 < items.size()) {
            const uint32_t* pNeedle = &items[warpIdx * table.itemSizeInU32];
            table.addAsWarp(pNeedle, warpIdx, outLocations[warpIdx]);
            //table.addAsWarp(pNeedle, warpIdx, tmp);
        }
    }
#endif
}
static __global__ void bulkFindAsWarp_kernel(DuplicateDetectionHashTable table, gsl::span<const uint32_t> items, gsl::span<uint32_t> outLocations)
{
#ifdef __CUDA_ARCH__
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < outLocations.size() + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx >= outLocations.size())
            break;

        const uint32_t* pNeedle = &items[warpIdx * table.itemSizeInU32];
        if (outLocations[warpIdx] == 0xFFFFFFFF)
            outLocations[warpIdx] = table.findAsWarp(pNeedle);
    }
#endif
}

TEST_CASE("DuplicateDetectionHashTable GPU construction (warp) & decoding (warp)", "[Atomic64HashTable]")
{
    constexpr size_t numUniqueItems = 51200;
    constexpr size_t duplicateFactor = 4;
    constexpr size_t numItems = numUniqueItems * duplicateFactor;
    constexpr size_t targetLoadFactor = 96;

    // const auto items = GENERATE(take(5, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));
    // const auto values = GENERATE(take(5, chunk(numItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    const auto uniqueItems = GENERATE(take(5, chunk(numUniqueItems * itemSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));
    std::vector<uint32_t> items(numItems * itemSizeInU32);
    std::minstd_rand rnd {};
    std::uniform_int_distribution<uint32_t> dist { 0, numUniqueItems - 1 };
    for (size_t i = 0; i < numItems; ++i) {
        const auto srcIdx = dist(rnd) * itemSizeInU32;
        std::memcpy(&items[i * itemSizeInU32], &uniqueItems[srcIdx], itemSizeInU32 * sizeof(uint32_t));
    }

    std::vector<uint32_t> values(numItems);
    std::iota(std::begin(values), std::end(values), 0);

    auto itemsGPU = StaticArray<uint32_t>::allocate("items", items, EMemoryType::GPU_Malloc);
    auto valuesGPU = StaticArray<uint32_t>::allocate("values", values, EMemoryType::GPU_Malloc);

    auto myHashTable = DuplicateDetectionHashTable::allocate(numItems / targetLoadFactor, numItems, itemSizeInU32);
    auto lookUpLocationsGPU = StaticArray<uint32_t>::allocate("out lookup locations", numItems, EMemoryType::GPU_Malloc);
    cudaMemset(lookUpLocationsGPU.data(), 0xFFFFFFFF, lookUpLocationsGPU.size_in_bytes());

    GPUTimingsManager timings {};
    {
        auto tmp = timings.timeScope("bulkAdd", nullptr);
        bulkAddAsWarp_kernel<<<32 * 1024, 64>>>(myHashTable, itemsGPU, numItems, lookUpLocationsGPU);
    }
    CUDA_CHECK_ERROR();

    {
        auto tmp = timings.timeScope("bulkFind", nullptr);
        bulkFindAsWarp_kernel<<<32 * 1024, 64>>>(myHashTable, itemsGPU, lookUpLocationsGPU);
    }
    CUDA_CHECK_ERROR();
    timings.print();

    auto lookUpLocations = lookUpLocationsGPU.copy_to_cpu();
    for (uint32_t i = 0; i < numItems; ++i) {
        const uint32_t* inItem = &items[i * itemSizeInU32];
        const uint32_t outItemIdx = lookUpLocations[i];
        const uint32_t* outItem = &items[outItemIdx * itemSizeInU32];
        CHECK(compareItem(inItem, outItem));
    }

    lookUpLocationsGPU.free();
    itemsGPU.free();
    valuesGPU.free();
    myHashTable.free();
}
