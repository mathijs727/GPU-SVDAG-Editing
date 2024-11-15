#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/ticket_board_hash_table.h"
#include "utils.h"

template <EMemoryType MemoryType>
TicketBoardHashTable<MemoryType> TicketBoardHashTable<MemoryType>::allocate(uint32_t numBuckets, uint32_t numReservedItems, uint32_t itemSizeInU32)
{
    PROFILE_FUNCTION();

    const auto numReservedSlabs = numBuckets + Utils::divideRoundUp(numReservedItems, items_per_slab);
    const auto slabSizeInU32 = 3 + items_per_slab * itemSizeInU32; // +3: slots mask, active mask, next ptr

    return TicketBoardHashTable(numBuckets, numReservedSlabs, slabSizeInU32, itemSizeInU32);
}

template class TicketBoardHashTable<EMemoryType::GPU_Malloc>;
template class TicketBoardHashTable<EMemoryType::GPU_Async>;
template class TicketBoardHashTable<EMemoryType::CPU>;