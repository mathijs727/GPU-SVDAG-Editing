#include "compact_acceleration_hash_table.h"
#include "utils.h"

template <EMemoryType MemoryType>
CompactAccelerationHashTable<MemoryType> CompactAccelerationHashTable<MemoryType>::allocate(uint32_t numBuckets, uint32_t numReservedItems, uint32_t itemSizeInU32)
{
    PROFILE_FUNCTION();

    const auto numReservedSlabs = numBuckets + Utils::divideRoundUp(numReservedItems, items_per_slab);
    const auto slabSizeInU32 = acceleration_hashes_size_in_u32 + 2 + items_per_slab * itemSizeInU32; // +2 for next ptr + active mask

    return CompactAccelerationHashTable(numBuckets, numReservedSlabs, slabSizeInU32, itemSizeInU32);
}

template class CompactAccelerationHashTable<EMemoryType::CPU>;
template class CompactAccelerationHashTable<EMemoryType::GPU_Async>;
template class CompactAccelerationHashTable<EMemoryType::GPU_Malloc>;
