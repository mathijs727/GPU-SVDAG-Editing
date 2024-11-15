#include <catch2/catch_all.hpp>
// ^^^ INCLUDE FIRST ^^^
#include "array.h"
#include "dags/my_gpu_dags/create_edit_svo.h"
#include "dags/my_gpu_dags/cub/cub_merge_sort.h"
#include "dags/my_gpu_dags/cub/cub_radix_sort.h"
#include "memory.h"
#include "timings.h"
#include "utils.h"
#include <random>
#include <vector>

using Item = typename IntermediateSVO::Leaf;

TEST_CASE("cubDeviceRadixSortKeys", "[MutableCUDABitStream][BitStream]")
{
    constexpr size_t numUniqueItems = Item::MaxSizeInU32 * 1024 * 128;
    constexpr size_t numItems = Item::MaxSizeInU32 * 4 * 1024 * 1024;

    std::uniform_int_distribution<uint32_t> dist;
    std::default_random_engine re { 12354 };

    std::vector<uint32_t> uniqueItems;
    for (size_t i = 0; i < numUniqueItems; ++i) {
        uniqueItems.push_back(dist(re));
    }

    std::vector<uint32_t> itemsCPU;
    std::vector<uint32_t> indicesCPU;
    for (size_t i = 0; i < numItems; ++i) {
        itemsCPU.push_back(uniqueItems[i % numUniqueItems]);
        indicesCPU.push_back(i);
    }
    std::shuffle(std::begin(itemsCPU), std::end(itemsCPU), re);

    auto itemsGPU = StaticArray<uint32_t>::allocate("itemsGPU", itemsCPU, EMemoryType::GPU_Malloc);
    auto indicesGPU = StaticArray<uint32_t>::allocate("indicesGPU", indicesCPU, EMemoryType::GPU_Malloc);

    GPUTimingsManager timings {};
    {
        auto tmp = timings.timeScope("cubDeviceRadixSortKeys", nullptr);
        for (int i = 0; i < Item::MaxSizeInU32; ++i) {
            cubDeviceRadixSortKeys<uint32_t>(itemsGPU.span(), indicesGPU.span(), nullptr);
        }
        CUDA_CHECK_ERROR();
    }
    timings.print();

    itemsGPU.free();
    indicesGPU.free();
    /*// Check that results are correct.
    std::vector<Item> controlItemsCPU(itemsCPU.size());
    std::vector<uint32_t> controlIndicesCPU(indicesCPU.size());
    cudaMemcpy(controlItemsCPU.data(), itemsGPU.data(), itemsGPU.size_in_bytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(controlIndicesCPU.data(), indicesGPU.data(), indicesGPU.size_in_bytes(), cudaMemcpyDeviceToHost);

    uint32_t numUniqueAfterSorting = 1; // Start with 1 unique (first item).
    for (uint32_t i = 1; i < controlItemsCPU.size(); ++i) {
        const auto& controlItem = controlItemsCPU[i];
        const auto& originalItem = itemsCPU[controlIndicesCPU[i]];
        REQUIRE(controlItem == originalItem);

        if (controlItem != controlItemsCPU[i - 1])
            ++numUniqueAfterSorting;
    }
    static_assert(numUniqueItems < numItems);
    REQUIRE(numUniqueAfterSorting == numUniqueItems);*/
}

TEST_CASE("cubMergeSortPairs", "[MutableCUDABitStream][BitStream]")
{
    constexpr size_t numUniqueItems = 1024 * 128;
    constexpr size_t numItems = 4 * 1024 * 1024;

    std::uniform_int_distribution<uint32_t> dist;
    std::default_random_engine re { 12354 };

    std::vector<Item> uniqueItems;
    for (size_t i = 0; i < numUniqueItems; ++i) {
        Item item {};
        item.size = item.MaxSizeInU32;
        for (int j = 0; j < item.MaxSizeInU32; ++j)
            item.padding[j] = dist(re);
        uniqueItems.push_back(item);
    }

    std::vector<Item> itemsCPU;
    std::vector<uint32_t> indicesCPU;
    for (size_t i = 0; i < numItems; ++i) {
        itemsCPU.push_back(uniqueItems[i % numUniqueItems]);
        indicesCPU.push_back(i);
    }
    std::shuffle(std::begin(itemsCPU), std::end(itemsCPU), re);

    auto itemsGPU = StaticArray<Item>::allocate("itemsGPU", itemsCPU, EMemoryType::GPU_Malloc);
    auto indicesGPU = StaticArray<uint32_t>::allocate("indicesGPU", indicesCPU, EMemoryType::GPU_Malloc);

    GPUTimingsManager timings {};
    {
        auto tmp = timings.timeScope("cubDeviceMergeSortPairs", nullptr);
        cubDeviceMergeSortPairs(itemsGPU.span(), indicesGPU.span(), nullptr);
        CUDA_CHECK_ERROR();
    }
    timings.print();

    // Check that results are correct.
    std::vector<Item> controlItemsCPU(itemsCPU.size());
    std::vector<uint32_t> controlIndicesCPU(indicesCPU.size());
    cudaMemcpy(controlItemsCPU.data(), itemsGPU.data(), itemsGPU.size_in_bytes(), cudaMemcpyDeviceToHost);
    cudaMemcpy(controlIndicesCPU.data(), indicesGPU.data(), indicesGPU.size_in_bytes(), cudaMemcpyDeviceToHost);
    itemsGPU.free();
    indicesGPU.free();

    uint32_t numUniqueAfterSorting = 1; // Start with 1 unique (first item).
    for (uint32_t i = 1; i < controlItemsCPU.size(); ++i) {
        const auto& controlItem = controlItemsCPU[i];
        const auto& originalItem = itemsCPU[controlIndicesCPU[i]];
        REQUIRE(controlItem == originalItem);

        if (controlItem != controlItemsCPU[i - 1])
            ++numUniqueAfterSorting;
    }
    static_assert(numUniqueItems < numItems);
    REQUIRE(numUniqueAfterSorting == numUniqueItems);
}
