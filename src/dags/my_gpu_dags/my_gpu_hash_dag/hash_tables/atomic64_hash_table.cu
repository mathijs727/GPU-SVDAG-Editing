#include "atomic64_hash_table.h"
#include "utils.h"

template <EMemoryType MemoryType>
Atomic64HashTable<MemoryType> Atomic64HashTable<MemoryType>::allocate(uint32_t numBuckets, uint32_t numReservedItems, uint32_t itemSizeInU32)
{
    PROFILE_FUNCTION();

    const auto numReservedSlabs = numBuckets + Utils::divideRoundUp(numReservedItems, items_per_slab);
    const auto slabSizeInU32 = 32 * itemSizeInU32;

    return Atomic64HashTable(numBuckets, numReservedSlabs, slabSizeInU32, itemSizeInU32);
}

template class Atomic64HashTable<EMemoryType::GPU_Malloc>;
template class Atomic64HashTable<EMemoryType::GPU_Async>;
template class Atomic64HashTable<EMemoryType::CPU>;
