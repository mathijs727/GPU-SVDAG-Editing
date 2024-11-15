#include "dynamic_slab_alloc.h"
#include "safe_cooperative_groups.h"
#include <algorithm>

static constexpr uint32_t maxWorkGroups = 8192 * 4;
static constexpr uint32_t warpsPerWorkGroup = 2;

template <size_t Alignment>
static __global__ void initSuperBlocksAsWarp(gsl::span<StaticArray<uint32_t, uint64_t, Alignment>> superBlocks, uint32_t bitmapPerSuperBlock)
{
#ifdef __CUDA_ARCH__
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    const auto threadRank = warp.thread_rank();

    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < superBlocks.size() + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx >= superBlocks.size())
            return;

        auto& superBlock = superBlocks[warpIdx];
        for (uint32_t offset = 0; offset < bitmapPerSuperBlock; offset += 32) {
            superBlock[offset + threadRank] = 0;
        }
    }
#endif
}

template <size_t Alignment, EMemoryType MemoryType>
DynamicSlabAllocator<Alignment, MemoryType> DynamicSlabAllocator<Alignment, MemoryType>::create(uint32_t initialNumItems, uint32_t memUnitSizeInU32)
{
    PROFILE_FUNCTION();

    const auto numSuperBlocks = Utils::divideRoundUp(initialNumItems, super_block_num_mem_units);

    DynamicSlabAllocator<Alignment, MemoryType> out;
    out.memUnitSizeInU32 = memUnitSizeInU32;
    out.superBlockSizeInU32 = num_bitmap_per_super_block + super_block_num_mem_units * memUnitSizeInU32;
    out.pNumAllocatedMemUnits = Memory::malloc<uint32_t>("DynamicSlabAllocator::pNumAllocatedMemUnits", sizeof(uint32_t), MemoryType);
    if constexpr (MemoryType == EMemoryType::CPU) {
        *out.pNumAllocatedMemUnits = 0;
    } else if (MemoryType == EMemoryType::GPU_Async) {
        CUDA_CHECKED_CALL cudaMemsetAsync(out.pNumAllocatedMemUnits, 0, sizeof(uint32_t), nullptr);
    } else {
        CUDA_CHECKED_CALL cudaMemset(out.pNumAllocatedMemUnits, 0, sizeof(uint32_t));
    }

    out.superBlocksCPU = DynamicArray<StaticArray<uint32_t, uint64_t, Alignment>>::allocate("DynamicSlabAllocator::superBlocksCPU", numSuperBlocks, EMemoryType::CPU);
    for (uint32_t i = 0; i < numSuperBlocks; ++i) {
        PROFILE_SCOPE("mallocSuperBlock");
        out.superBlocksCPU[i] = StaticArray<uint32_t, uint64_t, Alignment>::allocate("DynamicSlabAllocator::SuperBlock", out.superBlockSizeInU32, MemoryType);
#if ENABLE_CHECKS
        // Prevents compute-sanitizer --tool=initcheck from failing in the copy function (memcpy of uninitialized memory).
        if constexpr (MemoryType == EMemoryType::CPU) {
            memset(out.superBlocksCPU[i].data(), 0, out.superBlocksCPU[i].size_in_bytes());
        } else {
            cudaMemset(out.superBlocksCPU[i].data(), 0, out.superBlocksCPU[i].size_in_bytes());
        }
#endif
    }
    out.superBlocksGPU = out.superBlocksGPU.allocate("DynamicSlabAllocator::superBlocksGPU", out.superBlocksCPU, MemoryType);

#if 0
    for (uint32_t i = 0; i < numSuperBlocks; ++i) {
        if constexpr (MemoryType == EMemoryType::CPU) {
            // std::fill(std::begin(out.superBlocksCPU[i]), std::end(out.superBlocksCPU[i]), 0u);
            std::fill(std::begin(out.superBlocksCPU[i]), std::begin(out.superBlocksCPU[i]) + num_bitmap_per_super_block, 0u);
        } else if (MemoryType == EMemoryType::GPU_Async) {
            CUDA_CHECKED_CALL cudaMemsetAsync(out.superBlocksCPU[i].data(), 0, out.num_bitmap_per_super_block * sizeof(uint32_t), nullptr);
        } else {
            CUDA_CHECKED_CALL cudaMemset(out.superBlocksCPU[i].data(), 0, out.num_bitmap_per_super_block * sizeof(uint32_t));
        }
    }

#else
    const uint32_t numWorkGroups = std::min((numSuperBlocks + warpsPerWorkGroup - 1) / warpsPerWorkGroup, maxWorkGroups);
    if constexpr (MemoryType == EMemoryType::CPU) {
        for (auto& superBlock : out.superBlocksGPU)
            std::fill(std::begin(superBlock), std::begin(superBlock) + num_bitmap_per_super_block, 0u);
    } else {
        PROFILE_SCOPE("initSuperBlocks");
        initSuperBlocksAsWarp<Alignment><<<numWorkGroups, warpsPerWorkGroup * 32, 0, nullptr>>>(out.superBlocksGPU, num_bitmap_per_super_block);
        CUDA_CHECK_ERROR();
        cudaDeviceSynchronize();
    }

#endif

    cudaDeviceSynchronize();
    return out;
}

template <size_t Alignment, EMemoryType MemoryType>
void DynamicSlabAllocator<Alignment, MemoryType>::release()
{
    PROFILE_FUNCTION();
    for (auto superBlock : superBlocksCPU)
        superBlock.free();
    superBlocksCPU.free();
    superBlocksGPU.free();
    Memory::free(pNumAllocatedMemUnits);
}

template <size_t Alignment, EMemoryType MemoryType>
void DynamicSlabAllocator<Alignment, MemoryType>::reserveIfNecessary(uint32_t numExtraItems)
{
    CUDA_CHECK_ERROR();
    uint32_t numAllocatedMemUnits;
    if constexpr (MemoryType == EMemoryType::CPU) {
        numAllocatedMemUnits = *pNumAllocatedMemUnits;
    } else {
        CUDA_CHECKED_CALL cudaMemcpy(&numAllocatedMemUnits, pNumAllocatedMemUnits, sizeof(numAllocatedMemUnits), cudaMemcpyDefault);
    }

    const auto numRequiredSuperBlocks = Utils::divideRoundUp(numAllocatedMemUnits + numExtraItems, super_block_num_mem_units);
    if (numRequiredSuperBlocks <= superBlocksCPU.size())
        return;

    const auto numNewSuperBlocks = numRequiredSuperBlocks - superBlocksCPU.size();
    for (uint32_t i = 0; i < numNewSuperBlocks; ++i) {
        auto superBlock = StaticArray<uint32_t, uint64_t, Alignment>::allocate("DynamicSlabAllocator::SuperBlock", superBlockSizeInU32, MemoryType);
        if constexpr (MemoryType == EMemoryType::CPU) {
            std::fill(std::begin(superBlock), std::begin(superBlock) + num_bitmap_per_super_block, 0u);
        } else if constexpr (MemoryType == EMemoryType::GPU_Async) {
            CUDA_CHECKED_CALL cudaMemsetAsync(superBlock.data(), 0, num_bitmap_per_super_block * sizeof(uint32_t), nullptr);
        } else {
            CUDA_CHECKED_CALL cudaMemset(superBlock.data(), 0, num_bitmap_per_super_block * sizeof(uint32_t));
        }
        superBlocksCPU.push_back(superBlock);
    }

    using SuperBlockT = typename decltype(superBlocksCPU)::value_type;

    const auto oldSize = superBlocksGPU.size();
    superBlocksGPU.resize(superBlocksCPU.size());
    if constexpr (MemoryType == EMemoryType::CPU) {
        memcpy(&superBlocksGPU[oldSize], &superBlocksCPU[oldSize], (superBlocksGPU.size() - oldSize) * sizeof(SuperBlockT));
    } else if (MemoryType == EMemoryType::GPU_Async) {
        CUDA_CHECKED_CALL cudaMemcpyAsync(&superBlocksGPU[oldSize], &superBlocksCPU[oldSize], (superBlocksGPU.size() - oldSize) * sizeof(SuperBlockT), cudaMemcpyHostToDevice, nullptr);
    } else {
        CUDA_CHECKED_CALL cudaMemcpy(&superBlocksGPU[oldSize], &superBlocksCPU[oldSize], (superBlocksGPU.size() - oldSize) * sizeof(SuperBlockT), cudaMemcpyHostToDevice);
    }
    check(superBlocksCPU.size() == numRequiredSuperBlocks);
    check(superBlocksGPU.size() == numRequiredSuperBlocks);
}

template <size_t Alignment, EMemoryType MemoryType>
my_units::bytes DynamicSlabAllocator<Alignment, MemoryType>::memory_allocated() const
{
    return my_units::bytes(superBlocksGPU.size() * superBlockSizeInU32 * sizeof(uint32_t)) + superBlocksGPU.memory_allocated();
}

template <size_t Alignment, EMemoryType MemoryType>
my_units::bytes DynamicSlabAllocator<Alignment, MemoryType>::memory_used() const
{
#if CAPTURE_MEMORY_STATS_SLOW
    uint32_t numAllocatedMemUnits;
    if constexpr (MemoryType == EMemoryType::CPU) {
        numAllocatedMemUnits = *pNumAllocatedMemUnits;
    } else {
        cudaMemcpy(&numAllocatedMemUnits, pNumAllocatedMemUnits, sizeof(numAllocatedMemUnits), cudaMemcpyDeviceToHost);
    }
    return my_units::bytes((superBlocksGPU.size() * num_bitmap_per_super_block + numAllocatedMemUnits * memUnitSizeInU32) * sizeof(uint32_t)) + superBlocksGPU.memory_used();

#else
    return 0;
#endif
}

template <size_t Alignment, EMemoryType MemoryType>
void DynamicSlabAllocator<Alignment, MemoryType>::writeTo(BinaryWriter& writer) const
{
    writer.write(memUnitSizeInU32);
    writer.write(superBlockSizeInU32);
    writer.write(superBlocksCPU);
    writer.write(pNumAllocatedMemUnits, MemoryType);
}

template <size_t Alignment, EMemoryType MemoryType>
void DynamicSlabAllocator<Alignment, MemoryType>::readFrom(BinaryReader& reader)
{
    reader.read(memUnitSizeInU32);
    reader.read(superBlockSizeInU32);
    reader.read(superBlocksCPU);
    superBlocksGPU = superBlocksGPU.allocate("DynamicSlabAllocator::superBlocks", superBlocksCPU, MemoryType);
    reader.read(pNumAllocatedMemUnits, MemoryType);
}

template class DynamicSlabAllocator<128, EMemoryType::GPU_Managed>;
template class DynamicSlabAllocator<128, EMemoryType::GPU_Malloc>;
template class DynamicSlabAllocator<128, EMemoryType::GPU_Async>;
template class DynamicSlabAllocator<128, EMemoryType::CPU>;