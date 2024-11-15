#include "safe_cooperative_groups.h"
#include <catch2/catch_all.hpp>
// Include catch2 first.
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/dynamic_slab_alloc.h"
// Include dynamic_slab_alloc.h before any other code which may include typedefs.h
#include "cuda_helpers.h"

TEST_CASE("DynamicSlabAllocator allocate CPU", "[DynamicSlabAllocator]")
{
    static constexpr uint32_t memUnitSizeInU32 = 7;
    static constexpr uint32_t numSuperBlocks = 16;
    static constexpr uint32_t numMemUnits = numSuperBlocks * 32 * 32;

    const auto vs = GENERATE(take(10, chunk(numMemUnits * memUnitSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto slabAllocator = DynamicSlabAllocator<EMemoryType::CPU>::create(numMemUnits, memUnitSizeInU32);
    std::vector<uint32_t> ptrs;
    for (uint32_t i = 0; i < numMemUnits; ++i) {
        const uint32_t ptr = slabAllocator.allocate();
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j)
            pMemUnit[j] = vs[i * memUnitSizeInU32 + j];
        ptrs.push_back(ptr);
    }

    for (uint32_t i = 0; i < numMemUnits; ++i) {
        const uint32_t ptr = ptrs[i];
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j) {
            CHECK(pMemUnit[j] == vs[i * memUnitSizeInU32 + j]);
        }
    }

    slabAllocator.release();
}

static void __global__ decodePointers_kernel(DynamicSlabAllocator<EMemoryType::GPU_Malloc> slabAllocator, gsl::span<const uint32_t> inPointers, uint32_t memUnitSizeInU32, gsl::span<uint32_t> outMemUnits)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= inPointers.size())
        return;

    uint32_t* ptr = slabAllocator.decodePointer(inPointers[globalThreadIdx], __FILE__, __LINE__);
    for (uint32_t i = 0; i < memUnitSizeInU32; ++i)
        outMemUnits[globalThreadIdx * memUnitSizeInU32 + i] = ptr[i];
}

TEST_CASE("DynamicSlabAllocator::reserveIfNecessary() GPU", "[DynamicSlabAllocator]")
{
    static constexpr uint32_t memUnitSizeInU32 = 7;
    static constexpr uint32_t numSuperBlocks = 16;
    static constexpr uint32_t numMemUnits = numSuperBlocks * 32 * 32;

    auto slabAllocator = DynamicSlabAllocator<EMemoryType::GPU_Malloc>::create(numMemUnits, memUnitSizeInU32);
    slabAllocator.reserveIfNecessary(numMemUnits * 3);
    slabAllocator.release();
}

TEST_CASE("DynamicSlabAllocator copy to GPU", "[DynamicSlabAllocator]")
{
    static constexpr uint32_t memUnitSizeInU32 = 7;
    static constexpr uint32_t numSuperBlocks = 16;
    static constexpr uint32_t numMemUnits = numSuperBlocks * 32 * 32;

    const auto vs = GENERATE(take(10, chunk(numMemUnits * memUnitSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto slabAllocator = DynamicSlabAllocator<EMemoryType::CPU>::create(numMemUnits, memUnitSizeInU32);
    auto ptrs = StaticArray<uint32_t>::allocate("ptrs", numMemUnits, EMemoryType::GPU_Managed);
    for (uint32_t i = 0; i < numMemUnits; ++i) {
        const uint32_t ptr = ptrs[i] = slabAllocator.allocate();
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j)
            pMemUnit[j] = vs[i * memUnitSizeInU32 + j];
    }

    auto slabAllocatorGPU = slabAllocator.copy<EMemoryType::GPU_Malloc>();
    auto decodedMemUnits = StaticArray<uint32_t>::allocate("decodedMemUnits", vs, EMemoryType::GPU_Managed);
    decodePointers_kernel<<<computeNumWorkGroups(ptrs.size()), workGroupSize>>>(slabAllocatorGPU, ptrs, memUnitSizeInU32, decodedMemUnits);
    CUDA_CHECK_ERROR();
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < numMemUnits * memUnitSizeInU32; ++i) {
        CHECK(decodedMemUnits[i] == vs[i]);
    }

    ptrs.free();
    decodedMemUnits.free();
    slabAllocator.release();
    slabAllocatorGPU.release();
}

template <EMemoryType memoryType>
static void __global__ allocate_kernel(DynamicSlabAllocator<memoryType> slabAllocator, uint32_t memUnitSizeInU32, gsl::span<const uint32_t> inMemUnits, gsl::span<uint32_t> outPointers)
{
    const unsigned warpIdx = blockIdx.x;
    // slabAllocator.initAsWarp(warpIdx);
    const uint32_t ptr = slabAllocator.allocateAsWarp();
    if (threadIdx.x == 0) {
        outPointers[warpIdx] = ptr;
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t i = 0; i < memUnitSizeInU32; ++i) {
            pMemUnit[i] = inMemUnits[warpIdx * memUnitSizeInU32 + i];
        }
    }
    check(slabAllocator.isValidSlab(ptr, __FILE__, __LINE__));
}

TEST_CASE("DynamicSlabAllocator allocate GPU", "[DynamicSlabAllocator]")
{
    static constexpr uint32_t memUnitSizeInU32 = 3;
    // static constexpr uint32_t numSuperBlocks = 1024;
    static constexpr uint32_t numMemUnits = 32 * 1024;

    const auto vs = GENERATE(take(10, chunk(numMemUnits * memUnitSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));
    auto memUnits = StaticArray<uint32_t>::allocate("memUnits", vs, EMemoryType::GPU_Malloc);

    auto slabAllocator = DynamicSlabAllocator<EMemoryType::GPU_Malloc>::create(numMemUnits, memUnitSizeInU32);
    auto ptrs = StaticArray<uint32_t>::allocate("ptrs", numMemUnits, EMemoryType::GPU_Managed);

    allocate_kernel<<<numMemUnits, 32>>>(slabAllocator, memUnitSizeInU32, memUnits, ptrs);
    CUDA_CHECK_ERROR();
    cudaDeviceSynchronize();

    auto slabAllocatorCPU = slabAllocator.copy<EMemoryType::CPU>();
    for (uint32_t i = 0; i < numMemUnits; ++i) {
        const uint32_t ptr = ptrs[i];
        uint32_t* pMemUnit = slabAllocatorCPU.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j) {
            CHECK(pMemUnit[j] == vs[i * memUnitSizeInU32 + j]);
        }
    }

    memUnits.free();
    ptrs.free();
    slabAllocator.release();
    slabAllocatorCPU.release();
}

TEST_CASE("DynamicSlabAllocator free CPU", "[DynamicSlabAllocator]")
{
    static constexpr uint32_t memUnitSizeInU32 = 7;
    static constexpr uint32_t numSuperBlocks = 16;
    static constexpr uint32_t numMemUnits = numSuperBlocks * 32 * 32;
    static constexpr uint32_t numFreedMemUnits = numMemUnits / 2;

    const auto vs = GENERATE(take(10, chunk(numMemUnits * memUnitSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto slabAllocator = DynamicSlabAllocator<EMemoryType::CPU>::create(numMemUnits, memUnitSizeInU32);
    std::vector<uint32_t> ptrs;
    for (uint32_t i = 0; i < numMemUnits; ++i) {
        const uint32_t ptr = slabAllocator.allocate();
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j)
            pMemUnit[j] = vs[i * memUnitSizeInU32 + j];
        ptrs.push_back(ptr);
    }

    for (uint32_t i = 0; i < numFreedMemUnits; ++i) {
        check(slabAllocator.isValidSlab(ptrs[i], __FILE__, __LINE__));
        slabAllocator.free(ptrs[i]);
        check(!slabAllocator.isValidSlab(ptrs[i], __FILE__, __LINE__));
    }

    for (uint32_t i = 0; i < numFreedMemUnits; ++i) {
        const uint32_t ptr = slabAllocator.allocate();
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j)
            pMemUnit[j] = 123;
        ptrs[i] = ptr;
    }

    for (uint32_t i = 0; i < numMemUnits; ++i) {
        const uint32_t ptr = ptrs[i];
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j) {
            const auto expected = i < numFreedMemUnits ? 123 : vs[i * memUnitSizeInU32 + j];
            CHECK(pMemUnit[j] == expected);
        }
    }

    slabAllocator.release();
}

template <EMemoryType memoryType>
static void __global__ free_kernel(DynamicSlabAllocator<memoryType> slabAllocator, gsl::span<const uint32_t> inPointers)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= inPointers.size())
        return;

    slabAllocator.free(inPointers[globalThreadIdx]);
}

TEST_CASE("DynamicSlabAllocator free GPU", "[DynamicSlabAllocator]")
{
    static constexpr uint32_t memUnitSizeInU32 = 7;
    static constexpr uint32_t numSuperBlocks = 16;
    static constexpr uint32_t numMemUnits = numSuperBlocks * 32 * 32;
    static constexpr uint32_t numFreedMemUnits = numMemUnits / 2;

    const auto vs = GENERATE(take(10, chunk(numMemUnits * memUnitSizeInU32, random<uint32_t>(0, std::numeric_limits<uint32_t>::max()))));

    auto slabAllocator = DynamicSlabAllocator<EMemoryType::GPU_Managed>::create(numMemUnits, memUnitSizeInU32);
    auto ptrs = StaticArray<uint32_t>::allocate("ptrs", numMemUnits, EMemoryType::GPU_Managed);
    for (uint32_t i = 0; i < numMemUnits; ++i) {
        const uint32_t ptr = ptrs[i] = slabAllocator.allocate();
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j)
            pMemUnit[j] = vs[i * memUnitSizeInU32 + j];
    }

    free_kernel<<<computeNumWorkGroups(numFreedMemUnits), workGroupSize>>>(slabAllocator, ptrs);
    CUDA_CHECK_ERROR();
    cudaDeviceSynchronize();

    for (uint32_t i = 0; i < numFreedMemUnits; ++i) {
        const uint32_t ptr = slabAllocator.allocate();
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j)
            pMemUnit[j] = 123;
        ptrs[i] = ptr;
    }

    for (uint32_t i = 0; i < numMemUnits; ++i) {
        const uint32_t ptr = ptrs[i];
        uint32_t* pMemUnit = slabAllocator.decodePointer(ptr, __FILE__, __LINE__);
        for (uint32_t j = 0; j < memUnitSizeInU32; ++j) {
            const auto expected = i < numFreedMemUnits ? 123 : vs[i * memUnitSizeInU32 + j];
            CHECK(pMemUnit[j] == expected);
        }
    }

    slabAllocator.release();
    ptrs.free();
}
