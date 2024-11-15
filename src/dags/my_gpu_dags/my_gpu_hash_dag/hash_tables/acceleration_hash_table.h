#pragma once
#include "cuda_helpers.h"
#include "gpu_hash_table_base.h"
#include "typedefs.h"
#include "utils.h"
#include <gsl/gsl-lite.hpp>
#ifdef __CUDACC__
#include "safe_cooperative_groups.h"
#endif // __CUDACC__

#define OPTIMIZE_ITEM_SIZE_1 0

#if OPTIMIZE_ITEM_SIZE_1
#define ITEM_SIZE_1_COMPARISON_OPTIONAL itemSizeInU32 == 1
#else
#define ITEM_SIZE_1_COMPARISON_OPTIONAL false
#endif

template <EMemoryType MemoryType>
class AccelerationHashTable : public GpuHashTableBase<AccelerationHashTable, MemoryType> {
public:
    using Parent = GpuHashTableBase<AccelerationHashTable, MemoryType>;
    using Parent::itemSizeInU32;
    using Parent::not_found;

    using ElementDecoder = const uint32_t*;

public:
    using Parent::GpuHashTableBase;
    AccelerationHashTable() = default;
    static AccelerationHashTable allocate(uint32_t numBuckets, uint32_t numElements, uint32_t elementSizeInU32);

    template <uint32_t... Mask>
    HOST_DEVICE uint32_t add(const uint32_t* pItem)
    {
        const uint32_t accelerationHash = computeAccelerationHash(pItem);
        const uint32_t bucketIdx = this->computeBucket(pItem);

        INIT_ADD_VARIABLES()
        while (true) {
            if (slabIdx == next_sentinel)
                this->allocateNewSlabThread(bucketIdx, slabIdx, pSlab);

            static_assert(valid_slots_mask == 0xFFFFFFFE); // First lane does not participate.
            uint32_t dstLane = std::numeric_limits<uint32_t>::max();
            for (uint32_t lane = 1; lane < 32; ++lane) {
                if (atomicCAS_wrapper(&pSlab[lane], empty, accelerationHash) == empty) {
                    dstLane = lane;
                    break;
                }
            }

            if (dstLane < std::numeric_limits<uint32_t>::max()) {
#if OPTIMIZE_ITEM_SIZE_1
                if (itemSizeInU32 > 1) {
                    for (uint32_t j = 0; j < itemSizeInU32; ++j)
                        pSlab[32 + dstLane * itemSizeInU32 + j] = pItem[j];
                }
#else
                for (uint32_t j = 0; j < itemSizeInU32; ++j)
                    pSlab[32 + dstLane * itemSizeInU32 + j] = pItem[j];
#endif
                this->counterItemAdded();
                return this->encodePointer(bucketIdx, slabIdx, dstLane);
            }

            slabIdx = pSlab[next_pointer_offset_in_slab];
            if (slabIdx != next_sentinel)
                pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
        }

        checkAlways(false);
    }

#ifdef __CUDACC__
    DEVICE uint32_t addAsWarp(const uint32_t* pItem)
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        check(__activemask() == 0xFFFFFFFF);
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        const uint32_t item = pItem[threadRank] * (threadRank < itemSizeInU32);
        const auto accelerationHash = computeAccelerationHashAsWarp(item, threadRank);
        const auto bucketIdx = this->computeBucketAsWarp(pItem, item, threadRank);

        INIT_ADD_VARIABLES()
        while (true) {
            if (slabIdx == next_sentinel)
                this->allocateNewSlabWarp(threadRank, bucketIdx, slabIdx, pSlab);

            const uint32_t first = pSlab[threadRank];
            uint32_t emptyMask = warp.ballot(first == empty) & valid_slots_mask;
            if (emptyMask) {
                const uint32_t lane = __ffs(emptyMask) - 1u;
                uint32_t dstLane = std::numeric_limits<uint32_t>::max();
                if (lane == threadRank) {
                    check(accelerationHash != empty);
                    check(threadRank != next_pointer_offset_in_slab);
                    if (atomicCAS(&pSlab[threadRank], empty, accelerationHash) == empty)
                        dstLane = lane;
                }
                dstLane = warp.shfl(dstLane, lane);

                if (dstLane != std::numeric_limits<uint32_t>::max()) {
                    if (!(ITEM_SIZE_1_COMPARISON_OPTIONAL) && threadRank < itemSizeInU32)
                        pSlab[32 + dstLane * itemSizeInU32 + threadRank] = item;

                    this->counterItemAddedAsWarp(threadRank);
                    return this->encodePointer(bucketIdx, slabIdx, dstLane);
                }
            }
            slabIdx = pSlab[next_pointer_offset_in_slab];
            slabIdx = warp.shfl(slabIdx, 0); // Just to be sure...
            if (slabIdx != next_sentinel)
                pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
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

            const uint32_t first = srcSlab[threadRank];
            uint32_t emptyMask = warp.ballot(first == empty) & valid_slots_mask;
            if (emptyMask) {
                const uint32_t dstLane = __ffs(emptyMask) - 1u;
                bool inserted = false;
                if (dstLane == threadRank) {
                    check(srcAccelerationHash != empty);
                    check(threadRank != next_pointer_offset_in_slab);
                    if (atomicCAS(&srcSlab[threadRank], empty, srcAccelerationHash) == empty)
                        inserted = true;
                }

                if (warp.shfl(inserted, dstLane)) {
                    if (!(ITEM_SIZE_1_COMPARISON_OPTIONAL) && threadRank < itemSizeInU32)
                        srcSlab[32 + dstLane * itemSizeInU32 + threadRank] = srcItem[threadRank];

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

    template <uint32_t... Mask>
    HOST_DEVICE uint32_t find(const uint32_t* pNeedle) const
    {
        const auto accelerationHash = computeAccelerationHash(pNeedle);
        const auto bucketIdx = this->computeBucket(pNeedle);

        INIT_FIND_VARIABLES()
        while (slabIdx != next_sentinel) {
            for (uint32_t lane = 1; lane < 32; ++lane) {
                if (pSlab[lane] == accelerationHash) {
                    const uint32_t* pCompare = pSlab + 32 + lane * itemSizeInU32;
                    if (ITEM_SIZE_1_COMPARISON_OPTIONAL || Utils::compare_u32_array(pCompare, pNeedle, itemSizeInU32)) {
                        return this->encodePointer(bucketIdx, slabIdx, lane);
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
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        const uint32_t needle = pNeedle[threadRank] * (threadRank < itemSizeInU32);
        const auto accelerationHash = computeAccelerationHashAsWarp(needle, threadRank);
        const auto bucketIdx = this->computeBucketAsWarp(pNeedle, needle, threadRank);

        INIT_FIND_VARIABLES()
        while (slabIdx != next_sentinel) {
            static_assert(valid_slots_mask == 0xFFFFFFFE); // First lane does not participate.
            bool found = pSlab[threadRank] == accelerationHash && threadRank > 0;
            uint32_t foundMask = warp.ballot(found) & valid_slots_mask;
            if (foundMask) {
                const uint32_t* pCompare = pSlab + 32 + threadRank * itemSizeInU32;
                found &= Utils::compare_u32_array_varying_warp(pNeedle, pCompare, itemSizeInU32, threadRank, foundMask);

                foundMask = warp.ballot(found) & valid_slots_mask;
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

    // Copy of "findAsWarp" that returns how many slots potentially matched.
    DEVICE uint32_t findCountSlotsAsWarp(const uint32_t* pNeedle) const
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        const uint32_t needle = pNeedle[threadRank] * (threadRank < itemSizeInU32);
        const auto accelerationHash = computeAccelerationHashAsWarp(needle, threadRank);
        const auto bucketIdx = this->computeBucketAsWarp(pNeedle, needle, threadRank);

        INIT_FIND_VARIABLES()
        uint32_t numChecks = 0;
        while (slabIdx != next_sentinel) {
            static_assert(valid_slots_mask == 0xFFFFFFFE); // First lane does not participate.
            bool found = pSlab[threadRank] == accelerationHash && threadRank > 0;
            uint32_t foundMask = warp.ballot(found) & valid_slots_mask;
            if (foundMask) {
                numChecks += Utils::popc(foundMask);
                const uint32_t* pCompare = pSlab + 32 + threadRank * itemSizeInU32;
                found &= Utils::compare_u32_array_varying_warp(pNeedle, pCompare, itemSizeInU32, threadRank, foundMask);

                foundMask = warp.ballot(found) & valid_slots_mask;
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

    DEVICE uint32_t getPointerFromBucket(uint32_t bucketIdx, uint32_t itemIdx) const
    {
        return m_table[bucketIdx * this->m_slabSizeInU32 + itemIdx];
    }
    DEVICE uint32_t getPointerFromSlab(uint32_t slabIdx, uint32_t itemIdx) const
    {
        return m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__)[itemIdx];
    }
    DEVICE uint32_t findAsWarpHybrid(const uint32_t* pNeedle, bool toBeSearched) const
    {
        if (OPTIMIZE_ITEM_SIZE_1 && itemSizeInU32 == 1) {
            const uint32_t laneIdx = threadIdx.x & 31;
            uint32_t workQueue = 0, lastWorkQueue = 0;
            uint32_t next = base_node_pointer;
            const uint32_t key = toBeSearched ? pNeedle[0] : 0;

            const auto bucketIdx = toBeSearched ? this->computeBucket(pNeedle) : 0;

            uint32_t slabIdx, itemIdx = 0xFFFFFFFF;
            while (workQueue = __ballot_sync(0xFFFFFFFF, toBeSearched)) {
                next = (lastWorkQueue != workQueue) ? base_node_pointer : next;

                const uint32_t srcLane = __ffs(workQueue) - 1;
                const uint32_t srcBucket = __shfl_sync(0xFFFFFFFF, bucketIdx, srcLane);
                const uint32_t wantedKey = __shfl_sync(0xFFFFFFFF, key, srcLane);

                const uint32_t srcUnitData = next == base_node_pointer ? getPointerFromBucket(srcBucket, laneIdx) : getPointerFromSlab(next, laneIdx);
                int32_t foundLane = __ffs(__ballot_sync(0xFFFFFFFF, srcUnitData == wantedKey) & valid_slots_mask) - 1;
                if (foundLane < 0) { // not found
                    const uint32_t nextPtr = __shfl_sync(0xFFFFFFFF, srcUnitData, next_pointer_offset_in_slab);
                    if (nextPtr == next_sentinel) {
                        if (laneIdx == srcLane)
                            toBeSearched = false;
                    } else {
                        next = nextPtr;
                    }
                } else { // found the key
                    if (laneIdx == srcLane) {
                        slabIdx = next;
                        itemIdx = foundLane;
                        toBeSearched = false;
                    }
                }
                lastWorkQueue = workQueue;
            }

            if (itemIdx == 0xFFFFFFFF)
                return not_found;
            else
                return this->encodePointer(bucketIdx, slabIdx, itemIdx);
        } else {
            const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
            const auto threadRank = warp.thread_rank();

            // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
            const auto accelerationHash = toBeSearched ? computeAccelerationHash(pNeedle) : 0;
            const auto bucketIdx = toBeSearched ? this->computeBucket(pNeedle) : 0;

            bool isActive = toBeSearched;
            uint32_t itemIdx = 0xFFFFFFFF;
            INIT_FIND_VARIABLES()
            while (const uint32_t isActiveMask = warp.ballot(isActive)) {
                const auto srcLane = __ffs(isActiveMask) - 1u;
                const auto srcSlab = warp.shfl(pSlab, srcLane);
                const auto srcNeedle = warp.shfl(pNeedle, srcLane);
                const auto srcAccelerationHash = warp.shfl(accelerationHash, srcLane);

                static_assert(valid_slots_mask == 0xFFFFFFFE); // First lane does not participate.
                bool found = srcSlab[threadRank] == srcAccelerationHash && threadRank > 0;
                uint32_t foundMask = warp.ballot(found) & valid_slots_mask;
                if (foundMask) {
                    const uint32_t* pCompare = srcSlab + 32 + threadRank * itemSizeInU32;
                    found &= Utils::compare_u32_array_varying_warp(srcNeedle, pCompare, itemSizeInU32, threadRank, foundMask);

                    foundMask = warp.ballot(found) & valid_slots_mask;
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
    }
#endif

    HOST_DEVICE const uint32_t* decodePointer(uint32_t ptr) const
    {
        const auto [pSlab, itemIdx] = this->decodePointerImpl(ptr);
        return pSlab + 32 + itemIdx * itemSizeInU32;
    }

protected:
    static uint32_t numOccupiedSlotsInSlab(const uint32_t* pSlab)
    {
        uint32_t out = 0;
        for (uint32_t i = 1; i < 32; ++i) {
            out += pSlab[i] != empty;
        }
        return out;
    }

    HOST uint32_t freeItemsInSlab(uint32_t* pSlab, uint32_t inactiveMask)
    {
        inactiveMask &= valid_slots_mask;

        uint32_t counter = 0;
        for (uint32_t i = 0; i < 32; ++i) {
            if ((inactiveMask & (1u << i)) && pSlab[i] != empty) {
                pSlab[i] = empty;
                ++counter;
            }
        }
        return ++counter;
    }
#ifdef __CUDACC__
    DEVICE uint32_t freeItemsInSlabAsWarp(uint32_t* pSlab, uint32_t inactiveMask, uint32_t threadRank)
    {
        check(__activemask() == 0xFFFFFFFF);
        inactiveMask &= valid_slots_mask;

#if CAPTURE_MEMORY_STATS_SLOW
        bool cleared = false;
        if ((inactiveMask & (1u << threadRank)) && pSlab[threadRank] != empty) {
            pSlab[threadRank] = empty;
            cleared = true;
        }
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        return Utils::popc(warp.ballot(cleared));
#else
        if (inactiveMask & (1u << threadRank))
            pSlab[threadRank] = empty;
        return 0;
#endif
    }
#endif

private:
    // Compute the hash/value which will be used to accelerate the search process.
#ifdef __CUDACC__
    DEVICE uint32_t computeAccelerationHashAsWarp(uint32_t item, uint32_t threadRank) const
    {
#if OPTIMIZE_ITEM_SIZE_1
        return warp.shfl(item, 0);
#else
        uint32_t out = __reduce_xor_sync(0xFFFFFFFF, (32 - threadRank) * item); // Different than the main hash function.
        if (out == empty)
            ++out;
        return out;
#endif
    }
#endif
    HOST_DEVICE uint32_t computeAccelerationHash(const uint32_t* pItem) const
    {
#if OPTIMIZE_ITEM_SIZE_1
        return pItem[0];
#else
        uint32_t out = 0;
        for (uint32_t i = 0;  i < itemSizeInU32; ++i)
            out ^= (32 - i) * pItem[i];
        if (out == empty)
            ++out;
        return out;
#endif
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
    constexpr static uint32_t valid_slots_mask = 0xFFFFFFFE;

    using Parent::base_node_pointer;
    using Parent::next_sentinel;
    constexpr static uint32_t empty = next_sentinel;

    // ==== required by GpuHashTableBase ====
    constexpr static uint32_t next_pointer_offset_in_slab = 0; // First slot is reserved for next pointer.
    constexpr static uint32_t active_mask_offset_in_slab = 32; // Belongs to the unused first slot.
    constexpr static uint32_t slab_init_value = empty;
    constexpr static uint32_t slab_init_size = 32;
    // ======================================
};
