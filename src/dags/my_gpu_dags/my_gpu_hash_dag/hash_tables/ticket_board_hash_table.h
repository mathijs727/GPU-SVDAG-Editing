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
class TicketBoardHashTable : public GpuHashTableBase<TicketBoardHashTable, MemoryType> {
public:
    using Parent = GpuHashTableBase<TicketBoardHashTable, MemoryType>;
    using Parent::itemSizeInU32;
    using Parent::not_found;

    using ElementDecoder = const uint32_t*;

public:
    using Parent::GpuHashTableBase;
    TicketBoardHashTable() = default;
    static TicketBoardHashTable allocate(uint32_t numBuckets, uint32_t numReservedElements, uint32_t itemSizeInU32);

    HOST_DEVICE uint32_t add(const uint32_t* pItem)
    {
        const uint32_t bucketIdx = this->computeBucket(pItem);

        INIT_ADD_VARIABLES()
        while (true) {
            if (slabIdx == next_sentinel) {
                this->allocateNewSlabThread(bucketIdx, slabIdx, pSlab);
            }

            uint32_t& slotMask = pSlab[slot_mask_offset_in_slab];
            while (slotMask != 0xFFFFFFFF) {
#ifdef __CUDA_ARCH__
                const uint32_t itemIdx = __ffs(~slotMask) - 1u;
                const auto bitMask = 1u << itemIdx;
                const auto oldSlotMask = atomicOr(&slotMask, bitMask);
#else
                const uint32_t itemIdx = std::countr_one(slotMask);
                const auto bitMask = 1u << itemIdx;
                const auto oldSlotMask = slotMask;
                slotMask |= bitMask;
#endif

                if ((oldSlotMask & bitMask) == 0) {
                    // Successfully claimed a spot...
                    memcpy(&pSlab[items_offset_in_slab + itemIdx * itemSizeInU32], pItem, itemSizeInU32 * sizeof(uint32_t));
                    this->counterItemAdded();
                    return this->encodePointer(bucketIdx, slabIdx, itemIdx);
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
        const uint32_t bucketIdx = this->computeBucketAsWarp(pItem, item, threadRank);

        uint32_t itemIdx = not_found;
        INIT_ADD_VARIABLES()
        while (true) {
            if (slabIdx == next_sentinel)
                this->allocateNewSlabWarp(threadRank, bucketIdx, slabIdx, pSlab);

            static_assert(not_found > 32);
            if (threadRank == 0) {
                uint32_t& slotMask = pSlab[slot_mask_offset_in_slab];
                while (slotMask != 0xFFFFFFFF) {
                    itemIdx = __ffs(~slotMask) - 1u;
                    const auto bitMask = 1u << itemIdx;
                    const auto oldSlotMask = atomicOr(&slotMask, bitMask);
                    if ((oldSlotMask & bitMask) == 0) {
                        // Successfully claimed a spot...
                        this->counterItemAdded();
                        break;
                    } else {
                        itemIdx = not_found;
                    }
                }
            }
            itemIdx = warp.shfl(itemIdx, 0);
            if (itemIdx != not_found) {
                // Copy item into the bucket.
                if (threadRank < itemSizeInU32)
                    pSlab[items_offset_in_slab + itemIdx * itemSizeInU32 + threadRank] = pItem[threadRank];
                return this->encodePointer(bucketIdx, slabIdx, itemIdx);
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
        const uint32_t bucketIdx = isValid ? this->computeBucket(pItem) : 0;

        bool isActive = isValid;
        uint32_t itemIdx = not_found;
        INIT_ADD_VARIABLES()
        while (const uint32_t isActiveMask = warp.ballot(isActive)) {
            const auto srcLane = __ffs(isActiveMask) - 1u;
            uint32_t srcSlabIdx = warp.shfl(slabIdx, srcLane);
            uint32_t* srcSlab = warp.shfl(pSlab, srcLane);

            if (srcSlabIdx == next_sentinel) {
                const auto srcBucketIdx = warp.shfl(bucketIdx, srcLane);
                this->allocateNewSlabWarp(threadRank, srcBucketIdx, srcSlabIdx, srcSlab);
                if (threadRank == srcLane) {
                    slabIdx = srcSlabIdx;
                    pSlab = srcSlab;
                }
            }

            if (threadRank == srcLane) {
                uint32_t* pSlotMask = srcSlab + slot_mask_offset_in_slab;
                uint32_t slotMask = *pSlotMask;
                while (slotMask != 0xFFFFFFFF) {
                    itemIdx = __ffs(~slotMask) - 1u;
                    const auto bitMask = 1u << itemIdx;
                    slotMask = atomicOr(pSlotMask, bitMask);
                    if ((slotMask & bitMask) == 0) {
                        // Successfully claimed a spot...
                        this->counterItemAdded();
                        isActive = false;
                        check(itemIdx < 32);
                        break;
                    } else {
                        itemIdx = not_found;
                    }
                }
            }

            const auto dstLane = warp.shfl(itemIdx, srcLane);
            if (dstLane != not_found) {
                // Copy item into the bucket.
                const auto srcItem = warp.shfl(pItem, srcLane);
                if (threadRank < itemSizeInU32) {
                    auto tmp = srcItem[threadRank];
                    srcSlab[items_offset_in_slab + dstLane * itemSizeInU32 + threadRank] = tmp;
                }
            }

            if (threadRank == srcLane && itemIdx == not_found) {
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
#endif

    HOST_DEVICE uint32_t find(const uint32_t* pNeedle) const
    {
        const uint32_t bucketIdx = this->computeBucket(pNeedle);

        INIT_FIND_VARIABLES()
        while (slabIdx != next_sentinel) {
            for (uint32_t itemIdx = 0; itemIdx < 32; ++itemIdx) {
                if (pSlab[slot_mask_offset_in_slab] & (1u << itemIdx)) {
                    if (Utils::compare_u32_array(&pSlab[items_offset_in_slab + itemIdx * itemSizeInU32], pNeedle, itemSizeInU32)) {
                        return this->encodePointer(bucketIdx, slabIdx, itemIdx);
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
        const uint32_t bucketIdx = this->computeBucketAsWarp(pNeedle, needle, threadRank);

        INIT_FIND_VARIABLES()
        while (slabIdx != next_sentinel) {
            bool isEqual = false;
            if (pSlab[slot_mask_offset_in_slab] & (1u << threadRank))
                isEqual = Utils::compare_u32_array(&pSlab[items_offset_in_slab + threadRank * itemSizeInU32], pNeedle, itemSizeInU32);

            const auto threadMask = warp.ballot(isEqual) & pSlab[slot_mask_offset_in_slab];
            // check(threadMask == 0 || Utils::popc(threadMask) == 1);
            if (threadMask) {
                const auto itemIdx = __ffs(threadMask) - 1u;
                return this->encodePointer(bucketIdx, slabIdx, itemIdx);
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
        const auto threadRank = warp.thread_rank();

        // WARNING: will read slightly out-of-bounds, but sooo much faster than: threadRank < itemSizeInU32 ? pNeedle[threadRank] : 0;
        const auto bucketIdx = isValid ? this->computeBucket(pNeedle) : 0;

        bool isActive = isValid;
        uint32_t itemIdx = 0xFFFFFFFF;
        INIT_FIND_VARIABLES()
        while (const uint32_t isActiveMask = warp.ballot(isActive)) {
            const auto srcLane = __ffs(isActiveMask) - 1u;
            const auto srcSlab = warp.shfl(pSlab, srcLane);
            const auto srcNeedle = warp.shfl(pNeedle, srcLane);

            const uint32_t potentialSlots = srcSlab[slot_mask_offset_in_slab];
            const bool foundNonEmptySlot = potentialSlots & (1u << threadRank);
            if (potentialSlots) {
                bool found = false;
                if (foundNonEmptySlot) {
                    const uint32_t* pCompare = srcSlab + items_offset_in_slab + threadRank * itemSizeInU32;
                    if (Utils::compare_u32_array(pCompare, srcNeedle, itemSizeInU32)) {
                        found = true;
                    }
                }

                const auto foundMask = warp.ballot(found);
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
#endif

    HOST_DEVICE const uint32_t* decodePointer(uint32_t ptr) const
    {
        const auto [pSlab, itemIdx] = this->decodePointerImpl(ptr);
        return &pSlab[items_offset_in_slab + itemIdx * itemSizeInU32];
    }

protected:
    static uint32_t numOccupiedSlotsInSlab(const uint32_t* pSlab)
    {
        return Utils::popc(pSlab[slot_mask_offset_in_slab]);
    }

    HOST uint32_t freeItemsInSlab(uint32_t* pSlab, uint32_t inactiveMask)
    {
        const auto numCleared = Utils::popc(inactiveMask & pSlab[active_mask_offset_in_slab]);
        pSlab[active_mask_offset_in_slab] &= ~inactiveMask;
        return numCleared;
    }
#ifdef __CUDACC__
    DEVICE uint32_t freeItemsInSlabAsWarp(uint32_t* pSlab, uint32_t inactiveMask, uint32_t threadRank)
    {
#if CAPTURE_MEMORY_STATS_SLOW
        const auto numCleared = Utils::popc(inactiveMask & pSlab[active_mask_offset_in_slab]);
        __syncthreads();
        pSlab[active_mask_offset_in_slab] &= ~inactiveMask;
        return numCleared;
#else
        pSlab[active_mask_offset_in_slab] &= ~inactiveMask;
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
    using Parent::base_node_pointer;
    using Parent::next_sentinel;

    constexpr static uint32_t items_per_slab = 32;

    constexpr static uint32_t slot_mask_offset_in_slab = 0;
    constexpr static uint32_t items_offset_in_slab = 3;
    constexpr static uint32_t empty = 0;

    // ==== required by GpuHashTableBase ====
    constexpr static uint32_t next_pointer_offset_in_slab = 1;
    constexpr static uint32_t active_mask_offset_in_slab = 2;
    constexpr static uint32_t slab_init_value = empty;
    constexpr static uint32_t slab_init_size = 32;
    // ======================================
};
