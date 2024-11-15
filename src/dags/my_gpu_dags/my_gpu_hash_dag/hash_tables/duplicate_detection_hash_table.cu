#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/duplicate_detection_hash_table.h"

DuplicateDetectionHashTable DuplicateDetectionHashTable::allocate(uint32_t numBuckets, uint32_t numItems, uint32_t itemSizeInU32)
{
    PROFILE_FUNCTION();

    DuplicateDetectionHashTable out {};
    out.itemSizeInU32 = itemSizeInU32;
    out.slabSizeInU32 = 32 * (itemSizeInU32 + 1);
    out.table = StaticArray<uint32_t>::allocate("Atomic64HashTable::table", numBuckets, EMemoryType::GPU_Async);
    cudaMemsetAsync(out.table.data(), next_sentinel, out.table.size_in_bytes());
    // Give the allocator a couple extra slabs to reduce (infinite) waiting for the freelist.
    out.slabAllocator = decltype(out.slabAllocator)::create(1024 + numBuckets + Utils::divideRoundUp(numItems + numItems / 8, items_per_slab), out.slabSizeInU32);
    return out;
}

void DuplicateDetectionHashTable::free()
{
    PROFILE_FUNCTION();

    slabAllocator.release();
    table.free();
}

LinearFreeListAllocator LinearFreeListAllocator::create(uint32_t numItems, uint32_t itemSizeInU32)
{
    LinearFreeListAllocator out {};
    out.m_itemSizeInU32 = itemSizeInU32;
    out.m_numItems = numItems;
    out.m_pCurrentItem = Memory::malloc<uint32_t>("LinearFreeListAllocator pCurrentItem", sizeof(uint32_t), EMemoryType::GPU_Malloc);
    out.m_pData = Memory::malloc<uint32_t>("LinearFreeListAllocator pData", numItems * itemSizeInU32 * sizeof(uint32_t), EMemoryType::GPU_Malloc);
    cudaMemset(out.m_pCurrentItem, 0, sizeof(uint32_t));
    out.m_pFreeList = Memory::malloc<uint32_t>("LinearFreeListAllocator pFreeList", 32 * sizeof(uint32_t), EMemoryType::GPU_Malloc);
    cudaMemset(out.m_pFreeList, sentinel, 32 * sizeof(uint32_t));

    /* std::vector<uint32_t> v;
    v.resize(numItems * itemSizeInU32);
    for (uint32_t i = 0; i < numItems; ++i)
        v[i * itemSizeInU32] = (i + 1) * itemSizeInU32;
    v[(numItems - 1) * itemSizeInU32] = sentinel;
    cudaMemcpy(out.m_pData, v.data(), v.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(out.m_pFreeList, 0, sizeof(uint32_t)); */

    return out;
}

void LinearFreeListAllocator::release()
{
    Memory::free(m_pCurrentItem);
    Memory::free(m_pData);
    Memory::free(m_pFreeList);
}
