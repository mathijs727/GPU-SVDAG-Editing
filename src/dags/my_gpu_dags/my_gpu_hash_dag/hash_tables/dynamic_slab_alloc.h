#pragma once
#include "array.h"
#include "binary_reader.h"
#include "binary_writer.h"
#include "cuda_error_check.h"
#include "cuda_helpers_cpp.h"
#include "memory.h"
#include "safe_cooperative_groups.h"
#include "typedefs.h"
#include "utils.h"
#include <bit>
#include <cstdint>
#include <gsl/gsl-lite.hpp>
#include <random>

namespace dynamic_slab_alloc_impl {
// RNG used to visit random super blocks (CPU only).
// Use a global variable (with a fixed seed) to ensure determinism.
inline static thread_local std::minstd_rand0 re {};
}

// Implementation of SlabAlloc that supports (CPU directed) growing of the hash table pool.
// https://ieeexplore.ieee.org/abstract/document/8425196
//
// Code is very much inspired by the original implementation:
// https://github.com/owensgroup/SlabAlloc/blob/master/src/slab_alloc.cuh
template <size_t Alignment, EMemoryType MemoryType>
class DynamicSlabAllocator {
public:
    static DynamicSlabAllocator create(uint32_t initialNumItems, uint32_t memUnitSizeInU32);
    void release();

    my_units::bytes memory_allocated() const;
    my_units::bytes memory_used() const;

    template <EMemoryType NewMemoryType>
    DynamicSlabAllocator<Alignment, NewMemoryType> copy() const
    {
        PROFILE_FUNCTION();
        DynamicSlabAllocator<Alignment, NewMemoryType> out;
        out.memUnitSizeInU32 = memUnitSizeInU32;
        out.superBlockSizeInU32 = superBlockSizeInU32;
        out.superBlocksCPU = StaticArray<StaticArray<uint32_t, uint64_t, Alignment>>::allocate("DynamicSlabAllocator::superBlocksCPU", superBlocksCPU.size(), EMemoryType::CPU);
        for (uint32_t i = 0; i < superBlocksCPU.size(); ++i)
            out.superBlocksCPU[i] = superBlocksCPU[i].copy(NewMemoryType);
        out.superBlocksGPU = StaticArray<StaticArray<uint32_t, uint64_t, Alignment>>::allocate("DynamicSlabAllocator::superBlocksGPU", out.superBlocksCPU.span(), NewMemoryType);
        out.pNumAllocatedMemUnits = Memory::malloc<uint32_t>("DynamicSlabAllocator::pNumAllocatedMemUnits", sizeof(uint32_t), NewMemoryType);
        CUDA_CHECK_ERROR();
        if constexpr (MemoryType == EMemoryType::CPU && NewMemoryType == EMemoryType::CPU) {
            memcpy(out.pNumAllocatedMemUnits, this->pNumAllocatedMemUnits, sizeof(uint32_t));
        } else {
            CUDA_CHECKED_CALL cudaMemcpy(out.pNumAllocatedMemUnits, this->pNumAllocatedMemUnits, sizeof(uint32_t), cudaMemcpyDefault);
        }
        return out;
    }

    void reserveIfNecessary(uint32_t numExtraItems);

    HOST_DEVICE uint32_t allocate()
    {
        static constexpr uint32_t not_found = (uint32_t)-1;
#ifdef __CUDA_ARCH__
        State state { (uint32_t)superBlocksGPU.size(), threadIdx.x };
#else
        if constexpr (MemoryType == EMemoryType::CPU)
            reserveIfNecessary(super_block_num_mem_units);
        State state { (uint32_t)superBlocksGPU.size() };
#endif

#if ENABLE_CHECKS
        uint32_t numAttempts = 0;
#endif
        uint32_t memUnitIndex = not_found;
        while (memUnitIndex == not_found) {
            check(state.superBlockIndex < superBlocksGPU.size());
            auto superBlock = superBlocksGPU[state.superBlockIndex];

            for (uint32_t i = 0; i < num_bitmap_per_mem_block; ++i) {
                check(state.memoryBlockIndex * num_bitmap_per_mem_block + i < num_bitmap_per_super_block);
                check(state.memoryBlockIndex * num_bitmap_per_mem_block + i < superBlock.size());
                uint32_t& residentBitmap = superBlock[state.memoryBlockIndex * num_bitmap_per_mem_block + i];
#ifdef __CUDA_ARCH__
                const uint32_t firstEmptyBit = __ffs(~residentBitmap) - 1u;
                if (firstEmptyBit == 0xFFFFFFFF)
                    continue;
                const uint32_t mask = 1u << firstEmptyBit;
                const auto readBitmap = atomicOr(&residentBitmap, mask);
                if ((readBitmap & mask) == 0) {
                    // Successful insertion.
                    atomicAdd(pNumAllocatedMemUnits, 1);
                    memUnitIndex = i * 32 + firstEmptyBit;
                    break;
                }
#else
                const uint32_t firstEmptyBit = std::countr_one(residentBitmap);
                if (firstEmptyBit == 32)
                    continue;
                const uint32_t mask = 1u << firstEmptyBit;
                residentBitmap |= mask;
                (*pNumAllocatedMemUnits)++;
                memUnitIndex = i * 32 + firstEmptyBit;
                break;
#endif
            }

#if ENABLE_CHECKS
            if (numAttempts++ > 10000) {
                printf("superBlock = %u (out of %u); memUnitIdx = %u (out of %u)\n", state.superBlockIndex, (uint32_t)superBlocksGPU.size(), state.memoryBlockIndex, num_mem_blocks_per_super_block);
                // printf("Stuck allocating after %u loop iterations\n", numAttempts);
            }
#endif

            if (memUnitIndex == not_found)
                state.updateMemBlockIndex();
        }

        const uint32_t ptr = encodePointer(state.superBlockIndex, state.memoryBlockIndex, memUnitIndex);
#if defined(__CUDA_ARCH__) && ENABLE_CHECKS
        __threadfence(); // Make sure that the result of the atomicOr is visible to us.
#endif
        check(getSuperBlockIndex(ptr) == state.superBlockIndex);
        check(getMemoryBlockIndex(ptr) == state.memoryBlockIndex);
        check(getMemoryUnitIndex(ptr) == memUnitIndex);
        check(isValidSlab(ptr, __FILE__, __LINE__));
        return ptr;
    }

#ifdef __CUDACC__
    DEVICE uint32_t allocateAsWarp()
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        const auto threadRank = warp.thread_rank();
        const auto warpIdx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
        return allocateAsWarp(warpIdx);
    }
    DEVICE uint32_t allocateAsWarp(uint32_t warpIdx)
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        const auto threadRank = warp.thread_rank();
        // const auto warpIdx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
        check(__activemask() == 0xFFFFFFFF);
        check(warp.shfl(warpIdx, 0) == warpIdx);

        static constexpr uint32_t not_found = (uint32_t)-1;
        State state { (uint32_t)superBlocksGPU.size(), warpIdx };
        uint32_t memUnitIndex = not_found;
        uint32_t residentBitmap = superBlocksGPU[state.superBlockIndex][state.memoryBlockIndex * num_bitmap_per_mem_block + threadRank];
#if ENABLE_CHECKS
        int count = 0;
#endif
        while (memUnitIndex == not_found) {
            const uint32_t firstEmptyBit = __ffs(~residentBitmap) - 1u;
            const uint32_t sourceLane = __ffs(warp.ballot(firstEmptyBit != 0xFFFFFFFF)) - 1u;
            if (sourceLane == 0xFFFFFFFF) {
                state.updateMemBlockIndex();
#if ENABLE_CHECKS
                if (count++ > 100000) {
                    printf("STUCK ALLOCATING; memUnitSizeInU32 = %u; %u allocated out of %u superblock(s) (%u blocks)\n", memUnitSizeInU32, *pNumAllocatedMemUnits, (uint32_t)superBlocksGPU.size(), (uint32_t)superBlocksGPU.size() * num_mem_units_per_block * num_mem_blocks_per_super_block);
                }
#endif
                residentBitmap = superBlocksGPU[state.superBlockIndex][state.memoryBlockIndex * num_bitmap_per_mem_block + threadRank];
                continue;
            }

            if (threadRank == sourceLane) {
                const uint32_t mask = 1u << firstEmptyBit;
                const auto readBitmap = atomicOr(&superBlocksGPU[state.superBlockIndex][state.memoryBlockIndex * num_bitmap_per_mem_block + threadRank], mask);
                if ((readBitmap & mask) == 0) {
                    memUnitIndex = threadRank * 32 + firstEmptyBit;
                    atomicAdd(pNumAllocatedMemUnits, 1);
                } else {
                    residentBitmap = readBitmap;
                }
            }
            memUnitIndex = warp.shfl(memUnitIndex, sourceLane);
        }

        const uint32_t ptr = encodePointer(state.superBlockIndex, state.memoryBlockIndex, memUnitIndex);
        check(getSuperBlockIndex(ptr) == state.superBlockIndex);
        check(getMemoryBlockIndex(ptr) == state.memoryBlockIndex);
        check(getMemoryUnitIndex(ptr) == memUnitIndex);
        check(isValidSlab(ptr, __FILE__, __LINE__));
        return ptr;
    }
#endif // __CUDACC__

    HOST_DEVICE void free(uint32_t ptr)
    {
        // Super block and memory unit within super block.
        const auto superBlockIndex = getSuperBlockIndex(ptr);
        const auto memoryBlockIndex = getMemoryBlockIndex(ptr);
        const auto memoryUnitIndex = getMemoryUnitIndex(ptr);
        // Lane (uint32_t) and index within lane of the bitmap.
        const auto bitmapLane = memoryBlockIndex * num_bitmap_per_mem_block + (memoryUnitIndex >> 5);
        const auto bitmapIndex = memoryUnitIndex & 31;

        // Mask out the corresponding bit.
        const auto mask = ~(1u << bitmapIndex);
#ifdef __CUDA_ARCH__
        check(MemoryType != EMemoryType::CPU);
        atomicAnd(&superBlocksGPU[superBlockIndex][bitmapLane], mask);
        atomicSub(pNumAllocatedMemUnits, 1);
#else
        check(MemoryType != EMemoryType::GPU_Malloc);
        superBlocksGPU[superBlockIndex][bitmapLane] &= mask;
        *pNumAllocatedMemUnits -= 1;
#endif
    }

    HOST_DEVICE bool isValidSlab(uint32_t ptr, const char* file, int line) const
    {
#if defined(__CUDA_ARCH__) && ENABLE_CHECKS
        __threadfence();
#endif
        const auto superBlockIndex = getSuperBlockIndex(ptr);
        const auto memoryBlockIndex = getMemoryBlockIndex(ptr);
        const auto memoryUnitIndex = getMemoryUnitIndex(ptr);
        if (superBlockIndex >= superBlocksGPU.size()) {
            printf("%s a %i\n", file, line);
            return false;
        }

        const auto bitmapLane = memoryBlockIndex * num_bitmap_per_mem_block + (memoryUnitIndex >> 5);
        const auto bitmapIndex = memoryUnitIndex & 31;
        auto x = (superBlocksGPU[superBlockIndex][bitmapLane] & (1u << bitmapIndex));
        if (!x) {
            printf("%s b %i - %u\n", file, line, ptr);
            return false;
        }
        return x;
    }
    HOST_DEVICE uint32_t* decodePointer(uint32_t ptr, const char* file, int line)
    {
        check(isValidSlab(ptr, file, line));
        const auto superBlockIndex = getSuperBlockIndex(ptr);
        const auto memoryBlockIndex = getMemoryBlockIndex(ptr);
        const auto memoryUnitIndex = getMemoryUnitIndex(ptr);
        return &superBlocksGPU[superBlockIndex][num_bitmap_per_super_block + (memoryBlockIndex * num_mem_units_per_block + memoryUnitIndex) * memUnitSizeInU32];
    }
    HOST_DEVICE const uint32_t* decodePointer(uint32_t ptr, const char* file, int line) const
    {
        check(isValidSlab(ptr, file, line));
        const auto superBlockIndex = getSuperBlockIndex(ptr);
        const auto memoryBlockIndex = getMemoryBlockIndex(ptr);
        const auto memoryUnitIndex = getMemoryUnitIndex(ptr);
        return &superBlocksGPU[superBlockIndex][num_bitmap_per_super_block + (memoryBlockIndex * num_mem_units_per_block + memoryUnitIndex) * memUnitSizeInU32];
    }
    HOST_DEVICE bool isValidPointer(uint32_t) const { return true; }

    void writeTo(BinaryWriter& writer) const;
    void readFrom(BinaryReader& reader);

private:
    HOST_DEVICE static constexpr uint32_t encodePointer(uint32_t superBlockIndex, uint32_t memoryBlockIndex, uint32_t memoryUnitIndex)
    {
        uint32_t handle = 0;
        Utils::insert_bits<pointerSuperBlock_offset, pointerSuperBlock_size>(handle, superBlockIndex);
        Utils::insert_bits<pointerMemBlock_offset, pointerMemBlock_size>(handle, memoryBlockIndex);
        Utils::insert_bits<pointerMemUnit_offset, pointerMemUnit_size>(handle, memoryUnitIndex);
        return handle;
    }

    HOST_DEVICE static constexpr uint32_t getMemoryUnitIndex(uint32_t ptr) { return Utils::extract_bits<pointerMemUnit_offset, pointerMemUnit_size>(ptr); }
    HOST_DEVICE static constexpr uint32_t getMemoryBlockIndex(uint32_t ptr) { return Utils::extract_bits<pointerMemBlock_offset, pointerMemBlock_size>(ptr); }
    HOST_DEVICE static constexpr uint32_t getSuperBlockIndex(uint32_t ptr) { return Utils::extract_bits<pointerSuperBlock_offset, pointerSuperBlock_size>(ptr); }

private:
// Store state external from the slab allocator so that we don't run out of CUDA argument memory (limited to 4096 bytes).
#if 0 // Mirroring original implementation.
    struct State {
    public:
        uint32_t superBlockIndex;
        uint32_t memoryBlockIndex;
        uint32_t numAttempts = 0;

    public:
        HOST_DEVICE State(uint32_t numSuperBlocks, uint32_t warpIndex)
            : numSuperBlocks(numSuperBlocks)
            , warpIndex(warpIndex)
        {
            superBlockIndex = warpIndex % numSuperBlocks;
            memoryBlockIndex = (lgc_multiplier * warpIndex) >> (32 - log_num_mem_blocks_per_super_block);
        }
        HOST State(uint32_t numSuperBlocks)
            : State(numSuperBlocks, std::random_device()())
        {
        }

        HOST_DEVICE void updateMemBlockIndex()
        {
            ++numAttempts;
            ++superBlockIndex;
            superBlockIndex = (superBlockIndex == numSuperBlocks) ? 0 : superBlockIndex;
            //memoryBlockIndex = (lgc_multiplier * (warpIndex + numAttempts)) >> (32 - log_num_mem_blocks_per_super_block);
        }

    private:
        static constexpr uint32_t lgc_multiplier = 1103515245u;

        const uint32_t numSuperBlocks;
        const uint32_t warpIndex;
    };
#else // Improved state
    struct State {
    public:
        uint32_t superBlockIndex;
        uint32_t memoryBlockIndex;
        uint32_t lgcState;
        uint32_t numAttempts = 0;

    public:
        HOST_DEVICE State(uint32_t numSuperBlocks, uint32_t warpIndex)
            : numSuperBlocks(numSuperBlocks)
            , warpIndex(warpIndex)
        {
            static constexpr uint32_t memory_block_mask = (1u << log_num_mem_blocks_per_super_block) - 1u;

            superBlockIndex = warpIndex % numSuperBlocks;
            lgcState = (lgc_multiplier * warpIndex + lgc_increment);
            // Consider the upper 14 bits (16..30) as per ANSI C:
            // https://en.wikipedia.org/wiki/Linear_congruential_generator
            memoryBlockIndex = (lgcState >> 16) & memory_block_mask;
        }
        HOST State(uint32_t numSuperBlocks)
            : State(numSuperBlocks, (uint32_t)dynamic_slab_alloc_impl::re())
        {
        }

        HOST_DEVICE void updateMemBlockIndex()
        {
            static constexpr uint32_t memory_block_mask = (1u << log_num_mem_blocks_per_super_block) - 1u;

            ++numAttempts;
            ++superBlockIndex;
            superBlockIndex = (superBlockIndex == numSuperBlocks) ? 0 : superBlockIndex;
            lgcState = (lgcState + lgc_increment) * lgc_multiplier;
            // Consider the upper 14 bits (16..30) as per ANSI C:
            // https://en.wikipedia.org/wiki/Linear_congruential_generator
            memoryBlockIndex = (lgcState >> 16) & memory_block_mask;
        }

    private:
        // https://en.wikipedia.org/wiki/Linear_congruential_generator
        // Use the same values as glibc
        static constexpr uint32_t lgc_multiplier = 1103515245u;
        static constexpr uint32_t lgc_increment = 12345u;

        const uint32_t numSuperBlocks;
        const uint32_t warpIndex;
    };
#endif

private:
    template <size_t, EMemoryType>
    friend class DynamicSlabAllocator;

    static constexpr uint32_t num_bitmap_per_mem_block = 32;
    static constexpr uint32_t num_mem_units_per_block = num_bitmap_per_mem_block * 32;
    static constexpr uint32_t log_num_mem_blocks_per_super_block = 4;
    static constexpr uint32_t num_mem_blocks_per_super_block = 1u << log_num_mem_blocks_per_super_block;
    static constexpr uint32_t num_bitmap_per_super_block = num_mem_blocks_per_super_block * num_bitmap_per_mem_block;
    static constexpr uint32_t super_block_num_mem_units = num_mem_blocks_per_super_block * num_mem_units_per_block;

    static constexpr uint32_t pointerMemUnit_size = std::bit_width(num_mem_units_per_block - 1);
    static constexpr uint32_t pointerMemBlock_size = std::bit_width(num_mem_blocks_per_super_block - 1);
    static constexpr uint32_t pointerSuperBlock_size = 32u - pointerMemUnit_size - pointerMemBlock_size;

    static constexpr uint32_t pointerMemUnit_offset = 0u;
    static constexpr uint32_t pointerMemBlock_offset = pointerMemUnit_offset + pointerMemUnit_size;
    static constexpr uint32_t pointerSuperBlock_offset = pointerMemBlock_offset + pointerMemBlock_size;

    // Should be const but C++ is annoying and too much work right now.
    uint32_t memUnitSizeInU32;
    uint32_t superBlockSizeInU32;

    DynamicArray<StaticArray<uint32_t, uint64_t, Alignment>> superBlocksCPU;
    DynamicArray<StaticArray<uint32_t, uint64_t, Alignment>> superBlocksGPU;
    uint32_t* pNumAllocatedMemUnits;
};
