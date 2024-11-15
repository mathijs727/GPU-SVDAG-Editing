#include "acceleration_hash_table.h"
#include "utils.h"

template <EMemoryType MemoryType>
AccelerationHashTable<MemoryType> AccelerationHashTable<MemoryType>::allocate(uint32_t numBuckets, uint32_t numReservedItems, uint32_t itemSizeInU32)
{
    // 31 items per slab (1 slot reserved for bookkeeping).
    const auto numReservedSlabs = numBuckets + Utils::divideRoundUp(numReservedItems, 31);
#if OPTIMIZE_ITEM_SIZE_1
    const auto slabSizeInU32 = itemSizeInU32 == 1 ? 32 : 32 + 32 * itemSizeInU32;
#else
    const auto slabSizeInU32 = 32 + 32 * itemSizeInU32;
#endif

    return AccelerationHashTable<MemoryType>(numBuckets, numReservedSlabs, slabSizeInU32, itemSizeInU32);
}

template class AccelerationHashTable<EMemoryType::GPU_Malloc>;
template class AccelerationHashTable<EMemoryType::GPU_Async>;
template class AccelerationHashTable<EMemoryType::CPU>;
