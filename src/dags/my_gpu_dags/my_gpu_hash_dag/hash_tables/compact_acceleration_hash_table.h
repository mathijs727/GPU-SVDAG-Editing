#pragma once
#include "cuda_helpers.h"
#include "gpu_hash_table_base.h"
#include "typedefs.h"
#include "utils.h"
#include <gsl/gsl-lite.hpp>
#ifdef __CUDACC__
#include "safe_cooperative_groups.h"
#include <cooperative_groups/reduce.h>
#include <cuda/std/functional> // plus
#endif // __CUDACC__

template <EMemoryType MemoryType>
class CompactAccelerationHashTable : public GpuHashTableBase<CompactAccelerationHashTable, MemoryType> {
public:
    using Parent = GpuHashTableBase<CompactAccelerationHashTable, MemoryType>;
    using Parent::itemSizeInU32;
    using Parent::not_found;

    using ElementDecoder = const uint32_t*;

public:
    using Parent::GpuHashTableBase;
    CompactAccelerationHashTable() = default;
    static CompactAccelerationHashTable allocate(uint32_t numBuckets, uint32_t numElements, uint32_t elementSizeInU32);

    HOST_DEVICE uint32_t add(const uint32_t* pItem)
    {
        const uint32_t needleAccelerationHash = computeAccelerationHash(pItem);
        const uint32_t bucketIdx = this->computeBucket(pItem);

        INIT_ADD_VARIABLES()
        while (true) {
            if (slabIdx == next_sentinel)
                this->allocateNewSlabThread(bucketIdx, slabIdx, pSlab);

            for (uint32_t i = 0; i < acceleration_hashes_size_in_u32; ++i) {
                const uint32_t accelerationHashBlock = pSlab[acceleration_hashes_offset_in_slab + i];
                for (uint32_t j = 0; j < num_acceleration_hash_items_per_u32; ++j) {
                    const uint32_t k = j * num_acceleration_hash_bits;
                    if (((accelerationHashBlock >> k) & acceleration_hash_mask) == empty) {
                        static_assert(empty == 0);
                        check((needleAccelerationHash & acceleration_hash_mask) == needleAccelerationHash);
                        const uint32_t updatedHashBlock = accelerationHashBlock | (needleAccelerationHash << k);

                        const uint32_t indexInSlab = i * num_acceleration_hash_items_per_u32 + j;
                        if (::atomicCAS_wrapper(&pSlab[acceleration_hashes_offset_in_slab + i], accelerationHashBlock, updatedHashBlock) == accelerationHashBlock) {
                            for (uint32_t l = 0; l < itemSizeInU32; ++l)
                                pSlab[items_offset_in_slab + indexInSlab * itemSizeInU32 + l] = pItem[l];
                            this->counterItemAdded();
                            return this->encodePointer(bucketIdx, slabIdx, indexInSlab);
                        }
                    }
                }
            }

            slabIdx = pSlab[next_pointer_offset_in_slab];
            if (slabIdx != next_sentinel)
                pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
        }
    }
#ifdef __CUDACC__
    DEVICE uint32_t addAsWarp(const uint32_t* pItem)
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        check(__activemask() == 0xFFFFFFFF);
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        const uint32_t item = pItem[threadRank] * (threadRank < itemSizeInU32);
        const auto needleAccelerationHash = computeAccelerationHashAsWarp(item, threadRank);
        const auto bucketIdx = this->computeBucketAsWarp(pItem, item, threadRank);

        INIT_ADD_VARIABLES()
        while (true) {
            if (slabIdx == next_sentinel)
                this->allocateNewSlabWarp(threadRank, bucketIdx, slabIdx, pSlab);

            uint32_t accelerationHashBlock;
            if (threadRank < acceleration_hashes_size_in_u32)
                accelerationHashBlock = pSlab[acceleration_hashes_offset_in_slab + threadRank];

            static_assert(items_per_slab == 32);
            static_assert(std::popcount(num_acceleration_hash_bits) == 1); // Power of 2.
            const uint32_t hashBitOffsetFromStart = threadRank * num_acceleration_hash_bits;
            const uint32_t hashBlockIndex = hashBitOffsetFromStart >> 5;
            const uint32_t hashOffsetInBlock = hashBitOffsetFromStart & 31;
            check(hashBlockIndex < acceleration_hashes_size_in_u32);
            check(hashOffsetInBlock < 32);
            accelerationHashBlock = warp.shfl(accelerationHashBlock, hashBlockIndex);
            const uint32_t accelerationHash = (accelerationHashBlock >> hashOffsetInBlock) & acceleration_hash_mask;

            const uint32_t threadMask = warp.ballot(accelerationHash == empty);
            if (threadMask) {
                uint32_t indexInSlab = 0xFFFFFFFF;
                const uint32_t dstLane = __ffs(threadMask) - 1u;
                if (dstLane == threadRank) {
                    check((needleAccelerationHash & acceleration_hash_mask) == needleAccelerationHash);
                    check(accelerationHash == empty);
                    const auto updatedHashBlock = accelerationHashBlock | (needleAccelerationHash << hashOffsetInBlock);
                    // check(((pSlab[acceleration_hashes_offset_in_slab + hashBlockIndex] >> hashOffsetInBlock) & acceleration_hash_mask) == empty);
                    if (atomicCAS(&pSlab[acceleration_hashes_offset_in_slab + hashBlockIndex], accelerationHashBlock, updatedHashBlock) == accelerationHashBlock) {
                        indexInSlab = dstLane;
                        this->counterItemAdded();
                    }
                }
                indexInSlab = warp.shfl(indexInSlab, dstLane);

                if (indexInSlab != 0xFFFFFFFF) {
                    if (threadRank < itemSizeInU32)
                        pSlab[items_offset_in_slab + indexInSlab * itemSizeInU32 + threadRank] = item;
                    return this->encodePointer(bucketIdx, slabIdx, indexInSlab);
                }
            } else {
                slabIdx = pSlab[next_pointer_offset_in_slab];
                slabIdx = warp.shfl(slabIdx, 0); // Just to be sure...
                if (slabIdx != next_sentinel)
                    pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
            }
        }
    }
    DEVICE uint32_t addAsWarpHybrid(const uint32_t* pItem, bool isValid)
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        check(__activemask() == 0xFFFFFFFF);
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        uint32_t accelerationHash, bucketIdx = 0;
        if (isValid) {
            accelerationHash = computeAccelerationHash(pItem);
            bucketIdx = this->computeBucket(pItem);
        }

        bool isActive = isValid;
        uint32_t itemIdx;
        INIT_ADD_VARIABLES()
        while (const uint32_t isActiveMask = warp.ballot(isActive)) {
            const auto srcLane = __ffs(isActiveMask) - 1u;
            uint32_t srcSlabIdx = warp.shfl(slabIdx, srcLane);
            uint32_t* srcSlab = warp.shfl(pSlab, srcLane);
            const auto srcItem = warp.shfl(pItem, srcLane);
            const auto srcAccelerationHash = warp.shfl(accelerationHash, srcLane);

            if (srcSlabIdx == next_sentinel) {
                const auto srcBucketIdx = warp.shfl(bucketIdx, srcLane);
                this->allocateNewSlabWarp(threadRank, srcBucketIdx, srcSlabIdx, srcSlab);
                if (threadRank == srcLane) {
                    slabIdx = srcSlabIdx;
                    pSlab = srcSlab;
                }
            }

            uint32_t accelerationHashBlock;
            if (threadRank < acceleration_hashes_size_in_u32)
                accelerationHashBlock = srcSlab[acceleration_hashes_offset_in_slab + threadRank];

            static_assert(items_per_slab == 32);
            static_assert(std::popcount(num_acceleration_hash_bits) == 1); // Power of 2.
            const uint32_t hashBitOffsetFromStart = threadRank * num_acceleration_hash_bits;
            const uint32_t hashBlockIndex = hashBitOffsetFromStart >> 5;
            const uint32_t hashOffsetInBlock = hashBitOffsetFromStart & 31;
            check(hashBlockIndex < acceleration_hashes_size_in_u32);
            check(hashOffsetInBlock < 32);
            accelerationHashBlock = warp.shfl(accelerationHashBlock, hashBlockIndex);
            const uint32_t accelerationHashSlab = (accelerationHashBlock >> hashOffsetInBlock) & acceleration_hash_mask;

            uint32_t emptyMask = warp.ballot(accelerationHashSlab == empty);
            if (emptyMask) {
                const uint32_t dstLane = __ffs(emptyMask) - 1u;
                bool inserted = false;
                if (dstLane == threadRank) {
                    check((srcAccelerationHash & acceleration_hash_mask) == srcAccelerationHash);
                    check(accelerationHashSlab == empty);
                    const auto updatedHashBlock = accelerationHashBlock | (srcAccelerationHash << hashOffsetInBlock);
                    if (atomicCAS(&srcSlab[acceleration_hashes_offset_in_slab + hashBlockIndex], accelerationHashBlock, updatedHashBlock) == accelerationHashBlock) {
                        inserted = true;
                    }
                }

                if (warp.shfl(inserted, dstLane)) {
                    if (threadRank < itemSizeInU32)
                        srcSlab[items_offset_in_slab + dstLane * itemSizeInU32 + threadRank] = srcItem[threadRank];

                    this->counterItemAddedAsWarp(threadRank);
                    if (threadRank == srcLane) {
                        itemIdx = dstLane;
                        isActive = false;
                        continue;
                    }
                }
            } else if (threadRank == srcLane) {
                slabIdx = srcSlab[next_pointer_offset_in_slab];
                if (slabIdx != next_sentinel)
                    pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
            }
        }
        if (isValid)
            return this->encodePointer(bucketIdx, slabIdx, itemIdx);
        else
            return 0;
    }
#endif // __CUDACC__

    HOST_DEVICE uint32_t find(const uint32_t* pNeedle) const
    {
        const uint32_t needleAccelerationHash = computeAccelerationHash(pNeedle);
        const uint32_t bucketIdx = this->computeBucket(pNeedle);

        INIT_FIND_VARIABLES()
        while (slabIdx != next_sentinel) {
            for (uint32_t hashBlockIndex = 0; hashBlockIndex < acceleration_hashes_size_in_u32; ++hashBlockIndex) {
                const uint32_t accelerationHashBlock = pSlab[acceleration_hashes_offset_in_slab + hashBlockIndex];
                for (uint32_t hashInBlock = 0; hashInBlock < num_acceleration_hash_items_per_u32; ++hashInBlock) {
                    const uint32_t hashOffsetInBlock = hashInBlock * num_acceleration_hash_bits;
                    if (((accelerationHashBlock >> hashOffsetInBlock) & acceleration_hash_mask) == needleAccelerationHash) {
                        const uint32_t indexInSlab = hashBlockIndex * num_acceleration_hash_items_per_u32 + hashInBlock;
                        const uint32_t* pCompare = pSlab + items_offset_in_slab + indexInSlab * itemSizeInU32;
                        if (Utils::compare_u32_array(pCompare, pNeedle, itemSizeInU32))
                            return this->encodePointer(bucketIdx, slabIdx, indexInSlab);
                    }
                }
            }

            slabIdx = pSlab[next_pointer_offset_in_slab];
            if (slabIdx != next_sentinel)
                pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
        }

        return not_found;
    }
#ifdef __CUDACC__
    DEVICE uint32_t findAsWarp(const uint32_t* pNeedle) const
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        check(__activemask() == 0xFFFFFFFF);
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        const uint32_t needle = pNeedle[threadRank] * (threadRank < itemSizeInU32);
        const auto needleAccelerationHash = computeAccelerationHashAsWarp(needle, threadRank);
        const auto bucketIdx = this->computeBucketAsWarp(pNeedle, needle, threadRank);

        INIT_FIND_VARIABLES()
        while (slabIdx != next_sentinel) {
            uint32_t accelerationHashBlock;
            if (threadRank < acceleration_hashes_size_in_u32) {
                accelerationHashBlock = pSlab[acceleration_hashes_offset_in_slab + threadRank];
            }

            static_assert(items_per_slab == 32);
            static_assert(std::popcount(num_acceleration_hash_bits) == 1); // Power of 2.
            const uint32_t hashBitOffsetFromStart = threadRank * num_acceleration_hash_bits;
            const uint32_t hashBlockIndex = hashBitOffsetFromStart >> 5;
            const uint32_t hashOffsetInBlock = hashBitOffsetFromStart & 31;
            check(hashBlockIndex < acceleration_hashes_size_in_u32);
            check(hashOffsetInBlock < 32);
            accelerationHashBlock = warp.shfl(accelerationHashBlock, hashBlockIndex);
            const uint32_t accelerationHash = (accelerationHashBlock >> hashOffsetInBlock) & acceleration_hash_mask;

            bool found = accelerationHash == needleAccelerationHash;
            uint32_t foundMask = warp.ballot(found);
            if (foundMask) {
                const uint32_t* pCompare = pSlab + items_offset_in_slab + threadRank * itemSizeInU32;
                found &= Utils::compare_u32_array_varying_warp(pNeedle, pCompare, itemSizeInU32, threadRank, foundMask);

                foundMask = warp.ballot(found);
                if (foundMask) {
                    const auto itemIdx = __ffs(foundMask) - 1u;
                    return this->encodePointer(bucketIdx, slabIdx, itemIdx);
                }
            }

            slabIdx = pSlab[next_pointer_offset_in_slab];
            if (slabIdx != next_sentinel)
                pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
        }

        return not_found;
    }
    DEVICE uint32_t findAsWarpHybrid(const uint32_t* pNeedle, bool isActive)
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        const auto accelerationHash = isActive ? computeAccelerationHash(pNeedle) : 0;
        const auto bucketIdx = isActive ? this->computeBucket(pNeedle) : 0;

        uint32_t itemIdx = 0xFFFFFFFF;
        INIT_FIND_VARIABLES()
        while (const uint32_t isActiveMask = warp.ballot(isActive)) {
            const auto srcLane = __ffs(isActiveMask) - 1u;
            const auto srcSlab = warp.shfl(pSlab, srcLane);
            const auto srcNeedle = warp.shfl(pNeedle, srcLane);
            const auto srcAccelerationHash = warp.shfl(accelerationHash, srcLane);

            uint32_t accelerationHashBlock;
            if (threadRank < acceleration_hashes_size_in_u32) {
                accelerationHashBlock = srcSlab[acceleration_hashes_offset_in_slab + threadRank];
            }

            static_assert(items_per_slab == 32);
            static_assert(std::popcount(num_acceleration_hash_bits) == 1); // Power of 2.
            const uint32_t hashBitOffsetFromStart = threadRank * num_acceleration_hash_bits;
            const uint32_t hashBlockIndex = hashBitOffsetFromStart >> 5;
            const uint32_t hashOffsetInBlock = hashBitOffsetFromStart & 31;
            check(hashBlockIndex < acceleration_hashes_size_in_u32);
            check(hashOffsetInBlock < 32);
            accelerationHashBlock = warp.shfl(accelerationHashBlock, hashBlockIndex);
            const uint32_t accelerationHashSlab = (accelerationHashBlock >> hashOffsetInBlock) & acceleration_hash_mask;

            bool found = accelerationHashSlab == srcAccelerationHash;
            uint32_t foundMask = warp.ballot(found);
            if (foundMask) {
                const uint32_t* pCompare = srcSlab + items_offset_in_slab + threadRank * itemSizeInU32;
                found &= Utils::compare_u32_array_varying_warp(srcNeedle, pCompare, itemSizeInU32, threadRank, foundMask);

                foundMask = warp.ballot(found);
                if (foundMask) {
                    if (threadRank == srcLane) {
                        itemIdx = __ffs(foundMask) - 1u;
                        isActive = false;
                    }
                    continue;
                }
            }

            if (threadRank == srcLane) {
                slabIdx = pSlab[next_pointer_offset_in_slab];
                if (slabIdx != next_sentinel) {
                    pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
                } else {
                    isActive = false;
                }
            }
        }

        if (itemIdx == 0xFFFFFFFF)
            return not_found;
        else
            return this->encodePointer(bucketIdx, slabIdx, itemIdx);
    }

    // Copy of "findAsWarp" that returns how many slots potentially matched.
    DEVICE uint32_t findCountSlotsAsWarp(const uint32_t* pNeedle) const
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        check(__activemask() == 0xFFFFFFFF);
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        const uint32_t needle = pNeedle[threadRank] * (threadRank < itemSizeInU32);
        const auto needleAccelerationHash = computeAccelerationHashAsWarp(needle, threadRank);
        const auto bucketIdx = this->computeBucketAsWarp(pNeedle, needle, threadRank);

        INIT_FIND_VARIABLES()
        uint32_t numChecks = 0;
        while (slabIdx != next_sentinel) {
            uint32_t accelerationHashBlock;
            if (threadRank < acceleration_hashes_size_in_u32) {
                accelerationHashBlock = pSlab[acceleration_hashes_offset_in_slab + threadRank];
            }

            static_assert(items_per_slab == 32);
            static_assert(std::popcount(num_acceleration_hash_bits) == 1); // Power of 2.
            const uint32_t hashBitOffsetFromStart = threadRank * num_acceleration_hash_bits;
            const uint32_t hashBlockIndex = hashBitOffsetFromStart >> 5;
            const uint32_t hashOffsetInBlock = hashBitOffsetFromStart & 31;
            check(hashBlockIndex < acceleration_hashes_size_in_u32);
            check(hashOffsetInBlock < 32);
            accelerationHashBlock = warp.shfl(accelerationHashBlock, hashBlockIndex);
            const uint32_t accelerationHash = (accelerationHashBlock >> hashOffsetInBlock) & acceleration_hash_mask;

            bool found = accelerationHash == needleAccelerationHash;
            uint32_t foundMask = warp.ballot(found);
            if (foundMask) {
                numChecks += Utils::popc(foundMask);
                const uint32_t* pCompare = pSlab + items_offset_in_slab + threadRank * itemSizeInU32;
                found &= Utils::compare_u32_array_varying_warp(pNeedle, pCompare, itemSizeInU32, threadRank, foundMask);

                foundMask = warp.ballot(found);
                if (foundMask) {
                    const auto itemIdx = __ffs(foundMask) - 1u;
                    return numChecks;
                }
            }

            slabIdx = pSlab[next_pointer_offset_in_slab];
            if (slabIdx != next_sentinel)
                pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
        }

        return numChecks;
    }
#endif // __CUDACC__

    HOST_DEVICE const uint32_t* decodePointer(uint32_t ptr) const
    {
        const auto [pSlab, itemIdx] = this->decodePointerImpl(ptr);
        return &pSlab[items_offset_in_slab + itemIdx * itemSizeInU32];
    }

protected:
    static uint32_t numOccupiedSlotsInSlab(const uint32_t* pSlab)
    {
        uint32_t out = 0;
        for (uint32_t hashBlockIndex = 0; hashBlockIndex < acceleration_hashes_size_in_u32; ++hashBlockIndex) {
            const uint32_t accelerationHashBlock = pSlab[acceleration_hashes_offset_in_slab + hashBlockIndex];
            for (uint32_t hashInBlock = 0; hashInBlock < num_acceleration_hash_items_per_u32; ++hashInBlock) {
                const uint32_t hashOffsetInBlock = hashInBlock * num_acceleration_hash_bits;
                if (((accelerationHashBlock >> hashOffsetInBlock) & acceleration_hash_mask) != empty) {
                    ++out;
                }
            }
        }
        return out;
    }

    HOST uint32_t freeItemsInSlab(uint32_t* pSlab, uint32_t inactiveMask)
    {
        uint32_t numCleared = 0;
        for (uint32_t hashBlockIndex = 0; hashBlockIndex < acceleration_hashes_size_in_u32; ++hashBlockIndex) {
            uint32_t& accelerationHashBlock = pSlab[acceleration_hashes_offset_in_slab + hashBlockIndex];
            for (uint32_t hashInBlock = 0; hashInBlock < num_acceleration_hash_items_per_u32; ++hashInBlock) {
                const uint32_t hashOffsetInBlock = hashInBlock * num_acceleration_hash_bits;
                const uint32_t index = hashBlockIndex * num_acceleration_hash_items_per_u32 + hashInBlock;
                if (inactiveMask & (1u << index)) {
#if CAPTURE_MEMORY_STATS_SLOW
                    const auto tmp = ((accelerationHashBlock >> hashOffsetInBlock) & acceleration_hash_mask);
                    if (tmp != empty) {
                        ++numCleared;
                    }
#endif

                    static_assert(empty == 0);
                    const auto mask = acceleration_hash_mask << hashOffsetInBlock;
                    accelerationHashBlock &= ~mask;
                }
            }
        }
        return numCleared;
    }
#ifdef __CUDACC__
    DEVICE uint32_t freeItemsInSlabAsWarp(uint32_t* pSlab, uint32_t inactiveMask, uint32_t threadRank)
    {
        check(__activemask() == 0xFFFFFFFF);
        static_assert(items_per_slab == 32);

#if CAPTURE_MEMORY_STATS_SLOW
        uint32_t numCleared = 0;
#endif
        if (threadRank < acceleration_hashes_size_in_u32) {
            uint32_t mask = 0xFFFFFFFF;
            const uint32_t firstItemInThread = threadRank * num_acceleration_hash_items_per_u32;
            for (uint32_t i = 0; i < num_acceleration_hash_items_per_u32; ++i) {
                const uint32_t offset = (i * num_acceleration_hash_bits);
                if (inactiveMask & (1u << (firstItemInThread + i))) {
#if CAPTURE_MEMORY_STATS_SLOW
                    if (((pSlab[threadRank] >> offset) & acceleration_hash_mask) != empty) {
                        ++numCleared;
                    }
#endif
                    mask ^= acceleration_hash_mask << offset;
                }
            }

            pSlab[threadRank] &= mask;
        }

#if CAPTURE_MEMORY_STATS_SLOW
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        return cooperative_groups::reduce(warp, numCleared, cuda::std::plus<uint32_t>());
#else
        return 0;
#endif
    }
#endif

private:
    // Compute the hash/value which will be used to accelerate the search process.
#ifdef __CUDACC__
    DEVICE uint32_t computeAccelerationHashAsWarp(uint32_t item, uint32_t threadRank) const
    {
        uint32_t out = __reduce_xor_sync(0xFFFFFFFF, (32 - threadRank) * item); // Different than the main hash function.
        out &= acceleration_hash_mask;
        if (out == empty)
            out = 1;
        static_assert(empty != 1);
        return out;
    }
#endif
    HOST_DEVICE uint32_t computeAccelerationHash(const uint32_t* pItem) const
    {
        uint32_t out = 0;
        for (uint32_t i = 0;  i < itemSizeInU32; ++i)
            out ^= (32 - i) * pItem[i];
        out &= acceleration_hash_mask;
        if (out == empty)
            out = 1;
        static_assert(empty != 1);
        return out;
    }

private:
    template <template <EMemoryType> typename TChildBase, EMemoryType>
    friend class GpuHashTableBase;

    using Parent::m_numBuckets;
    using Parent::m_slabAllocator;
    using Parent::m_slabSizeInU32;
    using Parent::m_table;
#if CAPTURE_MEMORY_STATS_SLOW
    using Parent::m_pNumItems;
    using Parent::m_pNumSlabs;
#endif

private:
    constexpr static uint32_t log_items_per_slab = 5;
    constexpr static uint32_t items_per_slab = 1u << log_items_per_slab;

    constexpr static uint32_t num_acceleration_hash_bits = 8;
    constexpr static uint32_t acceleration_hash_mask = 0xFFFFFFFF >> (32 - num_acceleration_hash_bits);
    static_assert(32 % num_acceleration_hash_bits == 0);
    constexpr static uint32_t num_acceleration_hash_items_per_u32 = 32 / num_acceleration_hash_bits;
    constexpr static uint32_t acceleration_hashes_size_in_u32 = (num_acceleration_hash_bits * items_per_slab + 31) / 32;

    constexpr static uint32_t acceleration_hashes_offset_in_slab = 0;
    constexpr static uint32_t items_offset_in_slab = acceleration_hashes_size_in_u32 + 2; // +2 due to next pointer & activemask

    using Parent::base_node_pointer;
    using Parent::next_sentinel;
    constexpr static uint32_t empty = 0;

    // ==== required by GpuHashTableBase ====
    constexpr static uint32_t next_pointer_offset_in_slab = acceleration_hashes_size_in_u32 + 0;
    constexpr static uint32_t active_mask_offset_in_slab = acceleration_hashes_size_in_u32 + 1;
    constexpr static uint32_t slab_init_value = empty;
    constexpr static uint32_t slab_init_size = 32;
    static_assert(acceleration_hashes_size_in_u32 <= slab_init_size);
    // ======================================
};
