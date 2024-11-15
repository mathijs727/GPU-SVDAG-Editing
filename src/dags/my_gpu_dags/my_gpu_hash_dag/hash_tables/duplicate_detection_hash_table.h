#pragma once
#include "array.h"
#include "cuda_helpers.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/dynamic_slab_alloc.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/individual_chaining_hash_table.h"
#include "safe_cooperative_groups.h"
#include "typedefs.h"
#include <cuda.h>

// Lazily instantiated FreeList allocator.
// If the freelist is empty it will allocate from a linear allocator.
// If the linear allocator is full then the thread will spinlock.
// By creating a freelist on-the-fly we dodge the upfront cost of instantiating memory.
class LinearFreeListAllocator {
public:
    static LinearFreeListAllocator create(uint32_t numItems, uint32_t itemSizeInU32);
    void release();

    DEVICE uint32_t allocateAsWarp()
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

#if 0
        const uint32_t warpIdx = blockDim.x * blockIdx.x;
        const uint32_t threadRank = warp.thread_rank();

        uint32_t out = sentinel;
        while (true) {
            uint32_t headNode = m_pFreeList[threadRank];
            const uint32_t freeListMask = warp.ballot(headNode != sentinel);
            if (freeListMask) {
                const uint32_t bitOffset = warpIdx % __popc(freeListMask);
                const uint32_t selectedThread = __fns(freeListMask, 0, bitOffset + 1);
                // const uint32_t selectedThread = __ffs(freeListMask) - 1u;
                checkAlways(selectedThread < 32);
                if (threadRank == selectedThread)
                    checkAlways(headNode != sentinel);

                if (threadRank == selectedThread) {
                    uint32_t next;
                    do {
                        __threadfence();
                        out = headNode;
                        if (out == sentinel)
                            break;
                        next = m_pData[getHandleIndex(out)];
                        __threadfence();
                    } while ((headNode = atomicCAS(&m_pFreeList[selectedThread], out, next)) != out);
                }
                out = warp.shfl(out, selectedThread);
            } else {
                if (*m_pCurrentItem < m_totalSizeInU32) {
                    if (threadRank == 0) {
                        out = atomicAdd(m_pCurrentItem, m_itemSizeInU32);
                        if (out >= m_totalSizeInU32)
                            out = sentinel;
                    }
                    out = warp.shfl(out, 0);
                }
            }
            if (out != sentinel)
                return out;
        }
#else
        uint32_t out = sentinel;
        if (warp.thread_rank() == 0) {

            // Try to pop from freelist.
            uint32_t freeListIdx = (blockIdx.x * blockDim.x + threadIdx.x) & 31;

            while (out == sentinel) {
                // Read item from freelist.
                freeListIdx = (freeListIdx + 1) & 31;

                uint32_t headNode = m_pFreeList[freeListIdx];
                if (headNode == sentinel && *m_pCurrentItem < m_numItems) {
                    if (uint32_t nodeIndex = atomicAdd(m_pCurrentItem, 1); nodeIndex < m_numItems) {
                        out = nodeIndex;
                        break;
                    }
                }

                if (headNode == sentinel)
                    continue;

                __threadfence();
                const uint32_t next = m_pData[getHandleIndex(headNode) * m_itemSizeInU32];
                __threadfence();
                check(headNode != next);
                const uint32_t currentHeadNode = atomicCAS(&m_pFreeList[freeListIdx], headNode, next);
                if (headNode == currentHeadNode)
                    out = headNode;
            }

            // Check if the ABA problem might have occurred.
            // The use of tagged pointers should significantly reduce the chance of this happening.
            // https://en.wikipedia.org/wiki/ABA_problem
            // check(m_pData[getHandleIndex(out)] == next);

            check(out != sentinel);
        }
        out = warp.shfl(out, 0);
        check(out != sentinel);
        check(isValidPointer(out));
        return out;
#endif
    }
    DEVICE void free(uint32_t handle)
    {
        check(handle != sentinel);
        const uint32_t freeListIdx = (blockIdx.x * blockDim.x + threadIdx.x) & 31;

        check((handle & index_mask) < m_numItems);
        handle = incrementGeneration(handle);
        check((handle & index_mask) < m_numItems);
        const uint32_t indexInU32 = getHandleIndex(handle) * m_itemSizeInU32;
        check(indexInU32 < m_numItems * m_itemSizeInU32);
        do {
            __threadfence();
            m_pData[indexInU32] = m_pFreeList[freeListIdx];
            __threadfence();
        } while (atomicCAS(&m_pFreeList[freeListIdx], m_pData[indexInU32], handle) != m_pData[indexInU32]);
    }

    HOST_DEVICE bool isValidPointer(uint32_t handle) const
    {
        return getHandleIndex(handle) < m_numItems;
    }
    HOST_DEVICE uint32_t* decodePointer(uint32_t handle, const char*, int) { return &m_pData[getHandleIndex(handle) * m_itemSizeInU32]; }
    HOST_DEVICE const uint32_t* decodePointer(uint32_t handle, const char*, int) const { return &m_pData[getHandleIndex(handle) * m_itemSizeInU32]; }

private:
    static constexpr uint32_t sentinel = 0xFFFFFFFF;

    static constexpr uint32_t generation_bits = 5;
    static constexpr uint32_t index_bits = 32 - generation_bits;

    static constexpr uint32_t index_offset = 0;
    static constexpr uint32_t generation_offset = index_offset + index_bits;

    static constexpr uint32_t index_mask = ((1u << index_bits) - 1u) << index_offset;
    static constexpr uint32_t generation_mask = ((1u << generation_bits) - 1u) << generation_offset;

    HOST_DEVICE uint32_t getHandleIndex(uint32_t handle) const
    {
        return (handle & index_mask) >> index_offset;
    }
    HOST_DEVICE uint32_t incrementGeneration(uint32_t handle) const
    {
        const uint32_t index = (handle & index_mask) >> index_offset;
        const uint32_t generation = (handle & generation_mask) >> generation_offset;
        const uint32_t newShiftedGeneration = ((generation + 1u) << generation_offset) & generation_mask;
        const uint32_t out = (index << index_offset) | newShiftedGeneration;
        check(getHandleIndex(out) == getHandleIndex(handle));
        return out;
    }

private:
    // Linear allocator
    uint32_t* m_pData;
    uint32_t* m_pCurrentItem;
    uint32_t m_numItems;
    uint32_t m_itemSizeInU32;

    // Freelist
    uint32_t* m_pFreeList = nullptr;
};

class DuplicateDetectionHashTable {
public:
    static constexpr uint32_t not_found = 0xFFFFFFFF;
    uint32_t itemSizeInU32;

public:
    static DuplicateDetectionHashTable allocate(uint32_t numBuckets, uint32_t numItems, uint32_t itemSizeInU32);
    void free();

    __device__ void addAsWarp(uint32_t const* pItem, uint32_t value, uint32_t& existingValue)
    {
#ifdef __CUDA_ARCH__
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        check(__activemask() == 0xFFFFFFFF);
        const auto threadRank = warp.thread_rank();

        const uint32_t hash = Utils::murmurhash32xN(pItem, itemSizeInU32);
        // const uint32_t hash = Utils::xorHash32xN_warp(pItem[threadRank], itemSizeInU32, threadRank);
        const uint32_t bucketIdx = hash % table.size();
        const uint64_t itemFirstU64 = (uint64_t)pItem[0] | (((uint64_t)pItem[1]) << 32);
        if (itemFirstU64 == 0) // Not allowed! (would clash with "empty")
            return;
        check(itemFirstU64 != 0);

        uint32_t slabIdx = table[bucketIdx];
        uint32_t prevSlabIdx = slabIdx;
        while (true) {
            // Check if another thread has added a new slab to the head of the linked list.
            if (slabIdx == next_sentinel) {
                if (uint32_t newHead = table[bucketIdx]; newHead != prevSlabIdx)
                    slabIdx = newHead;
            }

            // Allocate a new slab if the current slab is full.
            if (slabIdx == next_sentinel) {
                slabIdx = slabAllocator.allocateAsWarp();
                check(slabIdx != next_sentinel);
                uint32_t* pSlab = slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
                static_assert(items_per_slab <= 32);
                pSlab[first_u64_offset_in_u32 + threadRank] = empty;
                pSlab[first_u64_offset_in_u32 + 32 + threadRank] = empty;
                pSlab[itemSizeInU32 * 32 + threadRank] = writing_lock_value;

                __threadfence();
                static_assert(next_pointer_offset_in_u32 >= 32 && next_pointer_offset_in_u32 < 64);
                if (threadRank == next_pointer_offset_in_u32 - 32) {
                    // Set to the previous head node.
                    // This may be outdated (other thread added to the linked list) but that is intended.
                    // If another thread already updated the linked list then we want to visit the slab that it has added.
                    pSlab[next_pointer_offset_in_u32] = prevSlabIdx;
                    __threadfence();

                    const uint32_t prevHead = atomicCAS(&table[bucketIdx], pSlab[next_pointer_offset_in_u32], slabIdx);
                    if (prevHead != pSlab[next_pointer_offset_in_u32]) {
                        slabAllocator.free(slabIdx);
                        slabIdx = prevHead;
                    }
                }
                slabIdx = warp.shfl(slabIdx, next_pointer_offset_in_u32);
            }

#if ENABLE_CHECKS
            __threadfence();
#endif
            check(slabAllocator.isValidPointer(slabIdx));
            uint32_t* pSlab = slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);

            uint64_t* pFirstItemsU64 = (uint64_t*)&pSlab[first_u64_offset_in_u32];
            const uint64_t firstItemsU64 = pFirstItemsU64[threadRank];
            static_assert(items_per_slab == 31);

            bool isMatchingSlot = (firstItemsU64 == itemFirstU64) && (threadRank < items_per_slab);
            if (isMatchingSlot) {
                // Other thread with matching first U64 took the slot. Wait for it to write
                // the rest of the item (+value) before checking whether we are the same.
                check(firstItemsU64 != empty);
                // Spinlock until other thread has written it's entire item.
                const uint32_t* pValueInTable = &pSlab[32 * itemSizeInU32 + threadRank];
                while ((*pValueInTable) == writing_lock_value) {
                    __threadfence();
                }
                static_assert(rest_of_items_offset_in_u32 == 64);
                for (uint32_t i = 2; i < itemSizeInU32; ++i) {
                    isMatchingSlot &= (pItem[i] == pSlab[32 * i + threadRank]);
                }
            }
            const uint32_t matchMask = warp.ballot(isMatchingSlot) & items_thread_mask;
            if (matchMask) {
                const uint32_t itemIdx = __ffs(matchMask) - 1u;
                existingValue = pSlab[32 * itemSizeInU32 + itemIdx];
                return;
            }

            const bool isEmptySlot = firstItemsU64 == empty;
            const uint32_t threadMask = warp.ballot(isEmptySlot) & items_thread_mask;
            if (threadMask) {
                uint32_t indexInSlab = 0xFFFFFFFF;
                const uint32_t dstLane = __ffs(threadMask) - 1u;
                if (dstLane == threadRank) {
                    if (atomicCAS_wrapper(&pFirstItemsU64[threadRank], empty, itemFirstU64) == empty) {
                        indexInSlab = dstLane;
                        for (uint32_t i = 2; i < itemSizeInU32; ++i) {
                            pSlab[32 * i + threadRank] = pItem[i];
                        }
                        check(value != writing_lock_value);
                        __threadfence(); // Make rest of the item visible before storing value.
                        pSlab[32 * itemSizeInU32 + threadRank] = value;
                        __threadfence(); // Force results to become visible to other threads.
                    }
                }
                if (warp.shfl(indexInSlab, dstLane) != 0xFFFFFFFF)
                    return;
            } else {
                prevSlabIdx = slabIdx;
                slabIdx = next_sentinel;
            }
        }
#endif // __CUDA_ARCH__
    }

    DEVICE uint32_t findAsWarp(const uint32_t* pNeedle) const
    {
#ifdef __CUDA_ARCH__
        check(__activemask() == 0xFFFFFFFF);
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        const auto threadRank = warp.thread_rank();

        // const uint32_t hash = Utils::xorHash32xN_warp(pNeedle[threadRank], itemSizeInU32, threadRank);
        const uint32_t hash = Utils::murmurhash32xN(pNeedle, itemSizeInU32);
        const uint32_t bucketIdx = hash % table.size();
        const uint64_t itemFirstU64 = (uint64_t)pNeedle[0] | (((uint64_t)pNeedle[1]) << 32);

        uint32_t slabIdx = table[bucketIdx];
        while (slabIdx != next_sentinel) {
            const uint32_t* pSlab = slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);

            const uint64_t* pFirstItemU64 = (const uint64_t*)&pSlab[first_u64_offset_in_u32];
            const uint64_t firstItemU64 = pFirstItemU64[threadRank];
            static_assert(items_per_slab == 31);

            bool isMatchingSlot = firstItemU64 == itemFirstU64 && threadRank < items_per_slab;
            if (warp.ballot(isMatchingSlot) & items_thread_mask) {
                for (uint32_t i = 2; i < itemSizeInU32; ++i) {
                    if (isMatchingSlot)
                        isMatchingSlot &= pNeedle[i] == pSlab[32 * i + threadRank];
                }

                const uint32_t threadMask = warp.ballot(isMatchingSlot) & items_thread_mask;
                if (threadMask)
                    return pSlab[32 * itemSizeInU32 + (__ffs(threadMask) - 1u)];
            }

            slabIdx = pSlab[next_pointer_offset_in_u32];
        }

        return not_found;
#endif // __CUDA_ARCH__
    }

private:
    uint32_t slabSizeInU32;

    DynamicSlabAllocator<128, EMemoryType::GPU_Async> slabAllocator;
    //   LinearAllocator<EMemoryType::GPU_Async> slabAllocator;
    //LinearFreeListAllocator slabAllocator;
    StaticArray<uint32_t> table;

    constexpr static uint32_t items_per_slab = 31;
    constexpr static uint32_t items_thread_mask = 0xFFFFFFFF >> (32 - items_per_slab);

    constexpr static uint32_t first_u64_offset_in_u32 = 0;
    constexpr static uint32_t next_pointer_offset_in_u32 = first_u64_offset_in_u32 + 31 * 2;
    constexpr static uint32_t rest_of_items_offset_in_u32 = 64;

    constexpr static uint32_t next_sentinel = 0xFFFFFFFF;
    constexpr static uint32_t writing_lock_value = 0xFFFFFFFF;
    constexpr static uint64_t empty = 0;
};
