#pragma once
#include "cuda_helpers.h"
#include "gpu_hash_table_base.h"
#include "utils.h"
#include <type_traits>
#ifdef __CUDACC__
#include "safe_cooperative_groups.h"
#include <cooperative_groups/reduce.h>
#endif // __CUDACC__

template <EMemoryType MemoryType>
class Atomic64HashTable : public GpuHashTableBase<Atomic64HashTable, MemoryType> {
public:
    using Parent = GpuHashTableBase<Atomic64HashTable, MemoryType>;
    using Parent::itemSizeInU32;
    using Parent::not_found;

    class ElementDecoder {
    public:
        HOST_DEVICE uint32_t operator[](size_t i) const
        {
            return (i < 2) ? pFirst[i] : pSecond[i - 2];
        }

    private:
        friend class Atomic64HashTable;
        const uint32_t* pFirst;
        const uint32_t* pSecond;
    };

public:
    using Parent::GpuHashTableBase;
    Atomic64HashTable() = default;
    static Atomic64HashTable allocate(uint32_t numBuckets, uint32_t numReservedElements, uint32_t itemSizeInU32);

    HOST_DEVICE uint32_t add(const uint32_t* pItem)
    {
        const uint64_t itemFirstU64 = (uint64_t)pItem[0] | (((uint64_t)pItem[1]) << 32);
        const uint32_t bucketIdx = this->computeBucket(pItem);
        check(itemFirstU64 != empty);

        INIT_ADD_VARIABLES()
        while (true) {
            if (slabIdx == next_sentinel)
                this->allocateNewSlabThread(bucketIdx, slabIdx, pSlab);

            uint64_t* pFirstItemsU64 = (uint64_t*)&pSlab[first_u64_offset_in_slab];
            for (uint32_t indexInSlab = 0; indexInSlab < items_per_slab; ++indexInSlab) {
                if (pFirstItemsU64[indexInSlab] == empty) {
                    if (::atomicCAS_wrapper(&pFirstItemsU64[indexInSlab], empty, itemFirstU64) == empty) {
                        for (uint32_t j = 0; j < itemSizeInU32 - 2; ++j)
                            pSlab[rest_of_items_offset_in_slab + indexInSlab * (itemSizeInU32 - 2) + j] = pItem[2 + j];
                        this->counterItemAdded();
                        return this->encodePointer(bucketIdx, slabIdx, indexInSlab);
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
#if !ENABLE_CHECKS
        const uint32_t item = pItem[threadRank] * (threadRank < itemSizeInU32);
#else
        const uint32_t item = threadRank < itemSizeInU32 ? pItem[threadRank] : 0;
#endif
        const uint64_t itemFirstU64 = (uint64_t)warp.shfl(item, 0) | (((uint64_t)warp.shfl(item, 1)) << 32);
        // const uint64_t itemFirstU64 = (uint64_t)pItem[0] | (((uint64_t)pItem[1]) << 32);
        const uint32_t bucketIdx = this->computeBucketAsWarp(pItem, item, threadRank);

        INIT_ADD_VARIABLES()
        while (true) {
            if (slabIdx == next_sentinel)
                this->allocateNewSlabWarp(threadRank, bucketIdx, slabIdx, pSlab);

            uint64_t* pFirstItemsU64 = (uint64_t*)&pSlab[first_u64_offset_in_slab];
            const uint64_t firstItemsU64 = pFirstItemsU64[threadRank];
            static_assert(items_per_slab == 31);

            const uint32_t threadMask = warp.ballot(firstItemsU64 == empty) & valid_slots_mask;
            // for (uint32_t i = 1; i <= Utils::popc(threadMask); ++i) {
            if (threadMask) {
                uint32_t indexInSlab = 0xFFFFFFFF;
                const uint32_t dstLane = __ffs(threadMask) - 1u;
                // const uint32_t dstLane = __fns(threadMask, 0, i);
                if (dstLane == threadRank) {
                    check(firstItemsU64 == empty);
                    if (::atomicCAS_wrapper(&pFirstItemsU64[threadRank], empty, itemFirstU64) == empty) {
                        indexInSlab = dstLane;
                    }
                }
                indexInSlab = warp.shfl(indexInSlab, dstLane);

                if (indexInSlab != 0xFFFFFFFF) {
                    if (threadRank < itemSizeInU32 - 2) {
                        const uint32_t value = pItem[2 + threadRank];
                        pSlab[rest_of_items_offset_in_slab + indexInSlab * (itemSizeInU32 - 2) + threadRank] = value;
                    }

                    this->counterItemAddedAsWarp(threadRank);
                    return this->encodePointer(bucketIdx, slabIdx, indexInSlab);
                }
            } else {
                slabIdx = pSlab[next_pointer_offset_in_slab];
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
        uint32_t bucketIdx = 0;
        uint64_t itemFirstU64;
        if (isValid) {
            itemFirstU64 = (uint64_t)pItem[0] | (((uint64_t)pItem[1]) << 32);
            bucketIdx = this->computeBucket(pItem);
        }

        bool isActive = isValid;
        uint32_t itemIdx;
        INIT_ADD_VARIABLES();
        while (const uint32_t isActiveMask = warp.ballot(isActive)) {

            const auto srcLane = __ffs(isActiveMask) - 1u;
            uint32_t srcSlabIdx = warp.shfl(slabIdx, srcLane);
            uint32_t* srcSlab = warp.shfl(pSlab, srcLane);
            const auto srcItem = warp.shfl(pItem, srcLane);
            const auto srcItemFirstU64 = warp.shfl(itemFirstU64, srcLane);

            if (srcSlabIdx == next_sentinel) {
                const auto srcBucketIdx = warp.shfl(bucketIdx, srcLane);
                this->allocateNewSlabWarp(threadRank, srcBucketIdx, srcSlabIdx, srcSlab);
                if (threadRank == srcLane) {
                    slabIdx = srcSlabIdx;
                    pSlab = srcSlab;
                }
            }

            uint64_t* pFirstItemsU64 = (uint64_t*)&srcSlab[first_u64_offset_in_slab];
            const uint64_t firstItemsU64 = pFirstItemsU64[threadRank];

            uint32_t emptyMask = warp.ballot(firstItemsU64 == empty) & valid_slots_mask;
            if (emptyMask) {
                const uint32_t dstLane = __ffs(emptyMask) - 1u;
                bool inserted = false;
                if (dstLane == threadRank) {
                    check(srcItemFirstU64 != empty);
                    if (atomicCAS_wrapper(&pFirstItemsU64[threadRank], empty, srcItemFirstU64) == empty)
                        inserted = true;
                }

                if (warp.shfl(inserted, dstLane)) {
                    if (threadRank < itemSizeInU32 - 2)
                        srcSlab[rest_of_items_offset_in_slab + dstLane * (itemSizeInU32 - 2) + threadRank] = srcItem[2 + threadRank];

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
        const uint64_t itemFirstU64 = (uint64_t)pNeedle[0] | (((uint64_t)pNeedle[1]) << 32);
        const uint32_t bucketIdx = this->computeBucket(pNeedle);

        INIT_FIND_VARIABLES()
        while (slabIdx != next_sentinel) {
            const uint64_t* pFirstItemsU64 = (const uint64_t*)&pSlab[first_u64_offset_in_slab];
            for (uint32_t indexInSlab = 0; indexInSlab < items_per_slab; ++indexInSlab) {
                if (pFirstItemsU64[indexInSlab] == itemFirstU64) {
                    const uint32_t* pCompare = &pSlab[rest_of_items_offset_in_slab + indexInSlab * (itemSizeInU32 - 2)];
                    if (Utils::compare_u32_array(pCompare, &pNeedle[2], itemSizeInU32 - 2)) {
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
#if !ENABLE_CHECKS
        const uint32_t needle = pNeedle[threadRank] * (threadRank < itemSizeInU32);
#else
        const uint32_t needle = threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
#endif
        const uint32_t bucketIdx = this->computeBucketAsWarp(pNeedle, needle, threadRank);
        const uint64_t itemFirstU64 = (uint64_t)warp.shfl(needle, 0) | (((uint64_t)warp.shfl(needle, 1)) << 32);

        INIT_FIND_VARIABLES()
        while (slabIdx != next_sentinel) {
            const uint64_t* pFirstItemU64 = (const uint64_t*)&pSlab[first_u64_offset_in_slab];
            const uint64_t firstItemU64 = pFirstItemU64[threadRank];
            static_assert(items_per_slab == 31);

            bool found = firstItemU64 == itemFirstU64 && threadRank < items_per_slab;
            uint32_t foundMask = warp.ballot(found) & valid_slots_mask;
            if (foundMask) {
                const uint32_t* pCompare = pSlab + rest_of_items_offset_in_slab + threadRank * (itemSizeInU32 - 2);
                found &= Utils::compare_u32_array_varying_warp(pNeedle + 2, pCompare, itemSizeInU32 - 2, threadRank, foundMask);
                foundMask = warp.ballot(found) & valid_slots_mask;
                check(Utils::popc(foundMask) <= 1);
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
    DEVICE uint32_t findAsWarpHybrid(const uint32_t* pNeedle, bool isValid) const
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        check(__activemask() == 0xFFFFFFFF);
        const auto threadRank = warp.thread_rank();

        uint32_t bucketIdx = 0;
        uint64_t itemFirstU64;
        if (isValid) {
            bucketIdx = this->computeBucket(pNeedle);
            itemFirstU64 = (uint64_t)pNeedle[0] | (((uint64_t)pNeedle[1]) << 32);
        }

        bool isActive = isValid;
        uint32_t itemIdx = 0xFFFFFFFF;
        INIT_FIND_VARIABLES()
        while (const uint32_t isActiveMask = warp.ballot(isActive)) {
            const auto srcLane = __ffs(isActiveMask) - 1u;
            const auto srcSlab = warp.shfl(pSlab, srcLane);
            const auto srcNeedle = warp.shfl(pNeedle, srcLane);
            const auto srcItemFirstU64 = warp.shfl(itemFirstU64, srcLane);

            const uint64_t* pFirstItemU64 = (const uint64_t*)&srcSlab[first_u64_offset_in_slab];
            const uint64_t firstItemU64 = pFirstItemU64[threadRank];
            static_assert(items_per_slab == 31);

            bool found = firstItemU64 == srcItemFirstU64 && threadRank < items_per_slab;
            uint32_t foundMask = warp.ballot(found) & valid_slots_mask;
            if (foundMask) {
                const uint32_t* pCompare = srcSlab + rest_of_items_offset_in_slab + threadRank * (itemSizeInU32 - 2);
                found &= Utils::compare_u32_array_varying_warp(srcNeedle + 2, pCompare, itemSizeInU32 - 2, threadRank, foundMask);

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
#endif // __CUDACC__

    void verifyNoDuplicates() const
    {
    }

    HOST_DEVICE ElementDecoder decodePointer(uint32_t ptr) const
    {
        const auto [pSlab, itemIdx] = this->decodePointerImpl(ptr);

        ElementDecoder out {};
        out.pFirst = &pSlab[first_u64_offset_in_slab + itemIdx * 2];
        out.pSecond = &pSlab[rest_of_items_offset_in_slab + itemIdx * (itemSizeInU32 - 2)];
        return out;
    }

protected:
    static uint32_t numOccupiedSlotsInSlab(const uint32_t* pSlab)
    {
        const uint64_t* pFirstItemsU64 = (uint64_t*)&pSlab[first_u64_offset_in_slab];
        uint32_t out = 0;
        for (uint32_t i = 0; i < 31; ++i) {
            out += pFirstItemsU64[i] != empty;
        }
        return out;
    }

    HOST uint32_t freeItemsInSlab(uint32_t* pSlab, uint32_t inactiveMask)
    {
        static_assert(items_per_slab <= 32);
        constexpr uint32_t items_mask = 0xFFFFFFFF >> (32 - items_per_slab);
        inactiveMask &= items_mask;

        uint32_t numCleared = 0;
        uint64_t* pFirstItemsU64 = (uint64_t*)&pSlab[first_u64_offset_in_slab];
        for (uint32_t i = 0; i < items_per_slab; ++i) {
            if ((inactiveMask & (1u << i)) && pFirstItemsU64[i] != empty) {
                pFirstItemsU64[i] = empty;
                ++numCleared;
            }
        }
        return numCleared;
    }
#ifdef __CUDACC__
    DEVICE uint32_t freeItemsInSlabAsWarp(uint32_t* pSlab, uint32_t inactiveMask, uint32_t threadRank)
    {
        check(__activemask() == 0xFFFFFFFF);

        static_assert(items_per_slab <= 32);
        constexpr uint32_t items_mask = 0xFFFFFFFF >> (32 - items_per_slab);
        inactiveMask &= items_mask;
        uint64_t* pFirstItemsU64 = (uint64_t*)&pSlab[first_u64_offset_in_slab];

#if CAPTURE_MEMORY_STATS_SLOW
        bool cleared = false;
        if ((inactiveMask & (1u << threadRank)) && (pFirstItemsU64[threadRank] != empty)) {
            pFirstItemsU64[threadRank] = empty;
            cleared = true;
        }
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        return Utils::popc(warp.ballot(cleared));
#else
        if (inactiveMask & (1u << threadRank))
            pFirstItemsU64[threadRank] = empty;
        return 0;
#endif
    }
#endif

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
    constexpr static uint32_t items_per_slab = 31;
    constexpr static uint32_t valid_slots_mask = 0xFFFFFFFF >> (32 - items_per_slab);

    constexpr static uint32_t first_u64_offset_in_slab = 0;
    constexpr static uint32_t rest_of_items_offset_in_slab = 64;

    using Parent::base_node_pointer;
    using Parent::next_sentinel;
    constexpr static uint64_t empty = 0;

    // ==== required by GpuHashTableBase ====
    constexpr static uint32_t next_pointer_offset_in_slab = 62;
    constexpr static uint32_t active_mask_offset_in_slab = 63;
    constexpr static uint32_t slab_init_value = empty;
    constexpr static uint32_t slab_init_size = 64;
    // ======================================
};
