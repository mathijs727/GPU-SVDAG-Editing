#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/atomic64_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/compact_acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/ticket_board_hash_table.h"
#include "gpu_hash_table_base.h"

static constexpr uint32_t maxWorkGroups = 8192 * 4;
static constexpr uint32_t warpsPerWorkGroup = 2;

template <uint32_t slab_init_size, uint32_t slab_init_value, uint32_t next_pointer_offset_in_slab, uint32_t next_sentinel>
static __global__ void initializeTableAsWarp(gsl::span<uint32_t> table, uint32_t numBuckets, uint32_t slabSizeInU32)
{
    const uint32_t warpIdx = threadIdx.x >> 5; // / 32
    const uint32_t threadRank = threadIdx.x & 31u;

    for (uint32_t bucketIdx = blockIdx.x * warpsPerWorkGroup + warpIdx; bucketIdx < numBuckets; bucketIdx += gridDim.x * warpsPerWorkGroup) {
        check(__activemask() == 0xFFFFFFFF);

        const uint32_t bucketStart = bucketIdx * slabSizeInU32;
        table[bucketStart + threadRank] = slab_init_value;
        if constexpr (slab_init_size > 32)
            table[bucketStart + 32 + threadRank] = slab_init_value;
        // We might have written to the next pointer memory location.
        // This will be overwritten but we need to make sure that this happens in the correct order.
        __syncwarp();
        table[bucketStart + next_pointer_offset_in_slab] = next_sentinel;
    }
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
GpuHashTableBase<TChildBase, MemoryType>::GpuHashTableBase(uint32_t numBuckets, uint32_t numReservedSlabs, uint32_t slabSizeInU32, uint32_t itemSizeInU32)
    : itemSizeInU32(itemSizeInU32)
{
    PROFILE_FUNCTION();

    m_slabSizeInU32 = slabSizeInU32;
    m_numBuckets = numBuckets;

#if HASH_TABLE_STORE_SLABS_IN_TABLE
    m_table = StaticArray<uint32_t, uint32_t, 128>::allocate("GpuHashTableBase::table", numBuckets * slabSizeInU32, MemoryType);
#if ENABLE_CHECKS
    // Prevents compute-sanitizer --tool=initcheck from failing in the copy function (memcpy of uninitialized memory).
    if constexpr (MemoryType == EMemoryType::CPU) {
        memset(m_table.data(), 0, m_table.size_in_bytes());
    } else {
        cudaMemset(m_table.data(), 0, m_table.size_in_bytes());
    }
#endif

    if constexpr (MemoryType == EMemoryType::CPU) {
        for (uint32_t bucketIdx = 0; bucketIdx < numBuckets; ++bucketIdx) {
            const uint32_t bucketStart = bucketIdx * slabSizeInU32;
            for (uint32_t i = 0; i < TChild::slab_init_size; ++i)
                m_table[bucketStart + i] = TChild::slab_init_value;
            m_table[bucketStart + TChild::next_pointer_offset_in_slab] = next_sentinel;
        }
    } else {
        initializeTableAsWarp<TChild::slab_init_size, TChild::slab_init_value, TChild::next_pointer_offset_in_slab, next_sentinel>
            <<<std::min(Utils::divideRoundUp(numBuckets, warpsPerWorkGroup), maxWorkGroups), warpsPerWorkGroup * 32>>>(
                m_table, numBuckets, slabSizeInU32);
        CUDA_CHECK_ERROR();
    }

#else // HASH_TABLE_STORE_SLABS_IN_TABLE
    m_table = StaticArray<uint32_t, uint32_t, 128>::allocate("GpuHashTableBase::table", numBuckets, MemoryType);
    if constexpr (MemoryType == EMemoryType::CPU) {
        memset(m_table.data(), next_sentinel, m_table.size_in_bytes());
    } else if constexpr (MemoryType == EMemoryType::GPU_Async) {
        cudaMemsetAsync(m_table.data(), next_sentinel, m_table.size_in_bytes());
    } else {
        cudaMemset(m_table.data(), next_sentinel, m_table.size_in_bytes());
    }
#endif // HASH_TABLE_STORE_SLABS_IN_TABLE

    m_slabAllocator = decltype(m_slabAllocator)::create(numReservedSlabs, slabSizeInU32);
#if CAPTURE_MEMORY_STATS_SLOW
    m_pNumItems = Memory::malloc<uint32_t>("GpuHashTableBase::pNumItems", sizeof(uint32_t), MemoryType);
    m_pNumSlabs = Memory::malloc<uint32_t>("GpuHashTableBase::pNumSlabs", sizeof(uint32_t), MemoryType);
    if constexpr (MemoryType == EMemoryType::CPU) {
        *m_pNumItems = 0;
#if HASH_TABLE_STORE_SLABS_IN_TABLE
        *m_pNumSlabs = numBuckets;
#else
        *m_pNumSlabs = 0;
#endif
    } else {
        cudaMemset(m_pNumItems, 0, sizeof(uint32_t));
#if HASH_TABLE_STORE_SLABS_IN_TABLE
        cudaMemcpy(m_pNumSlabs, &numBuckets, sizeof(uint32_t), cudaMemcpyHostToDevice);
#else
        cudaMemset(m_pNumSlabs, 0, sizeof(uint32_t));
#endif
    }
#endif
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
void GpuHashTableBase<TChildBase, MemoryType>::free()
{
    m_slabAllocator.release();
    m_table.free();
#if CAPTURE_MEMORY_STATS_SLOW
    Memory::free(m_pNumItems);
    Memory::free(m_pNumSlabs);
#endif
}
template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
void GpuHashTableBase<TChildBase, MemoryType>::reserveIfNecessary(uint32_t numAdditionalItems)
{
    const uint32_t numFullSlabs = Utils::divideRoundUp(numAdditionalItems, 31);
    const uint32_t numPartialSlabs = numBuckets();
    const uint32_t numTotalSlabs = std::min(numFullSlabs + numPartialSlabs, numAdditionalItems);
    m_slabAllocator.reserveIfNecessary(numTotalSlabs);
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
void GpuHashTableBase<TChildBase, MemoryType>::writeTo(BinaryWriter& writer) const
{
    writer.write(itemSizeInU32);
    writer.write(m_slabSizeInU32);
    writer.write(m_numBuckets);
    writer.write(m_table);
    writer.write(m_slabAllocator);
    uint32_t numItems = 0, numSlabs = 0;
#if CAPTURE_MEMORY_STATS_SLOW
    if constexpr (MemoryType == EMemoryType::CPU) {
        numItems = *m_pNumItems;
        numSlabs = *m_pNumSlabs;
    } else {
        cudaMemcpy(&numItems, m_pNumItems, sizeof(numItems), cudaMemcpyDeviceToHost);
        cudaMemcpy(&numSlabs, m_pNumSlabs, sizeof(numSlabs), cudaMemcpyDeviceToHost);
    }
#endif
    writer.write(numItems);
    writer.write(numSlabs);
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
void GpuHashTableBase<TChildBase, MemoryType>::readFrom(BinaryReader& reader)
{
    reader.read(itemSizeInU32);
    reader.read(m_slabSizeInU32);
    reader.read(m_numBuckets);
    reader.read(m_table);
    reader.read(m_slabAllocator);

    uint32_t numItems, numSlabs;
    reader.read(numItems);
    reader.read(numSlabs);

#if CAPTURE_MEMORY_STATS_SLOW
    m_pNumItems = Memory::malloc<uint32_t>("Atomic64HashTable::pNumItems", sizeof(uint32_t), MemoryType);
    m_pNumSlabs = Memory::malloc<uint32_t>("Atomic64HashTable::pNumSlabs", sizeof(uint32_t), MemoryType);
    if constexpr (MemoryType == EMemoryType::CPU) {
        *m_pNumItems = numItems;
        *m_pNumSlabs = numSlabs;
    } else {
        cudaMemcpy(m_pNumItems, &numItems, sizeof(numItems), cudaMemcpyHostToDevice);
        cudaMemcpy(m_pNumSlabs, &numSlabs, sizeof(numSlabs), cudaMemcpyHostToDevice);
    }
#endif
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
my_units::bytes GpuHashTableBase<TChildBase, MemoryType>::memory_allocated() const
{
    return m_slabAllocator.memory_allocated() + m_table.memory_allocated();
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
my_units::bytes GpuHashTableBase<TChildBase, MemoryType>::memory_used_by_items() const
{
    my_units::bytes out = 0;
#if CAPTURE_MEMORY_STATS_SLOW
    const my_units::bytes itemSizeInBytes { itemSizeInU32 * sizeof(uint32_t) };
    if constexpr (MemoryType == EMemoryType::CPU) {
        out += (*m_pNumItems) * itemSizeInBytes;
    } else {
        uint32_t numItems;
        cudaMemcpy(&numItems, m_pNumItems, sizeof(numItems), cudaMemcpyDeviceToHost);
        out += numItems * itemSizeInBytes;
    }
#endif
    return out;
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
my_units::bytes GpuHashTableBase<TChildBase, MemoryType>::memory_used_by_slabs() const
{
    my_units::bytes out = 0;
#if CAPTURE_MEMORY_STATS_SLOW
    const my_units::bytes slabSizeInBytes { m_slabSizeInU32 * sizeof(uint32_t) };
    if constexpr (MemoryType == EMemoryType::CPU) {
        checkAlways((*m_pNumSlabs) >= numBuckets());
        out += (*m_pNumSlabs) * slabSizeInBytes;
    } else {
        uint32_t numSlabs;
        cudaMemcpy(&numSlabs, m_pNumSlabs, sizeof(numSlabs), cudaMemcpyDeviceToHost);
#if HASH_TABLE_STORE_SLABS_IN_TABLE
        checkAlways(numSlabs >= numBuckets());
#endif
        out += numSlabs * slabSizeInBytes;
    }
#endif
    return out;
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
double GpuHashTableBase<TChildBase, MemoryType>::currentLoadFactor() const
{
    if constexpr (MemoryType == EMemoryType::CPU) {
        uint32_t numElements = 0;
        for (uint32_t bucketIdx = 0; bucketIdx < this->numBuckets(); ++bucketIdx) {
            uint32_t const* pSlab = &m_table[bucketIdx * m_slabSizeInU32];
            uint32_t slabIdx = base_node_pointer;
            while (slabIdx != next_sentinel) {
                numElements += TChild::numOccupiedSlotsInSlab(pSlab);

                slabIdx = pSlab[TChild::next_pointer_offset_in_slab];
                if (slabIdx != next_sentinel)
                    pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
            }
        }
        return (double)numElements / (double)this->numBuckets();
    } else {
        return 0.0;
    }
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
__host__ __device__ void GpuHashTableBase<TChildBase, MemoryType>::_clearActiveFlagsBucket(uint32_t bucketIdx)
{
    uint32_t slabIdx = base_node_pointer;
    uint32_t* pSlab = &m_table[bucketIdx * m_slabSizeInU32];
    while (slabIdx != next_sentinel) {
        pSlab[TChild::active_mask_offset_in_slab] = 0;

        slabIdx = pSlab[TChild::next_pointer_offset_in_slab];
        if (slabIdx != next_sentinel)
            pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
    }
}

template <typename T>
static __global__ void clearActiveFlags_kernel(T table, uint32_t numBuckets)
{
    const auto globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < numBuckets)
        table._clearActiveFlagsBucket(globalThreadIdx);
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
void GpuHashTableBase<TChildBase, MemoryType>::clearActiveFlags()
{
    if constexpr (MemoryType == EMemoryType::CPU) {
        for (uint32_t bucketIdx = 0; bucketIdx < numBuckets(); ++bucketIdx)
            _clearActiveFlagsBucket(bucketIdx);
    } else {
        clearActiveFlags_kernel<<<computeNumWorkGroups(numBuckets()), workGroupSize>>>(*this, numBuckets());
        CUDA_CHECK_ERROR();
    }
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
__device__ void GpuHashTableBase<TChildBase, MemoryType>::_freeInactiveItemsBucket_warp(uint32_t bucketIdx)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto threadRank = warp.thread_rank();
    uint32_t numItemsFreed = 0, numSlabsFreed = 0;

    uint32_t slabIdx = base_node_pointer;
    uint32_t* pSlab = &m_table[bucketIdx * m_slabSizeInU32];
    uint32_t* pPrevNextPtr = nullptr;
    while (slabIdx != next_sentinel) {
        const uint32_t activeMask = pSlab[TChild::active_mask_offset_in_slab];
        numItemsFreed += ((TChild*)this)->freeItemsInSlabAsWarp(pSlab, ~activeMask, threadRank);

        if (threadRank == 0) {
            const uint32_t nextSlabIdx = pSlab[TChild::next_pointer_offset_in_slab];
            if (activeMask == 0 && slabIdx != base_node_pointer) {
                m_slabAllocator.free(slabIdx);
                *pPrevNextPtr = nextSlabIdx;
                ++numSlabsFreed;
            } else {
                pPrevNextPtr = &pSlab[TChild::next_pointer_offset_in_slab];
            }
            slabIdx = nextSlabIdx;
        }

        slabIdx = warp.shfl(slabIdx, 0);
        if (slabIdx != next_sentinel) {
            pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
        }
    }

#if CAPTURE_MEMORY_STATS_SLOW
    if (threadRank == 0) {
        atomicSub(m_pNumItems, numItemsFreed);
        atomicSub(m_pNumSlabs, numSlabsFreed);
    }
#endif
}

template <typename T>
static __global__ void freeInactiveItems_kernel(T table, uint32_t numBuckets)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();

    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < numBuckets; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx >= numBuckets)
            return;
        table._freeInactiveItemsBucket_warp(warpIdx);
    }
}

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
uint32_t GpuHashTableBase<TChildBase, MemoryType>::freeInactiveItems()
{
    if (numBuckets() == 0)
        return 0;

    if constexpr (MemoryType == EMemoryType::CPU) {
        uint32_t numItemsFreed = 0, numSlabsFreed = 0;
        for (uint32_t bucketIdx = 0; bucketIdx < numBuckets(); ++bucketIdx) {
            uint32_t slabIdx = base_node_pointer;
            uint32_t* pSlab = &m_table[bucketIdx * m_slabSizeInU32];
            uint32_t* pPrevNextPtr = nullptr;
            while (slabIdx != next_sentinel) {
                const uint32_t activeMask = pSlab[TChild::active_mask_offset_in_slab];
                numItemsFreed += ((TChild*)this)->freeItemsInSlab(pSlab, ~activeMask);
                const uint32_t nextSlabIdx = pSlab[TChild::next_pointer_offset_in_slab];
                if (activeMask == 0 && slabIdx != base_node_pointer) {
                    m_slabAllocator.free(slabIdx);
                    *pPrevNextPtr = nextSlabIdx;
                    ++numSlabsFreed;
                } else {
                    pPrevNextPtr = &pSlab[TChild::next_pointer_offset_in_slab];
                }

                slabIdx = nextSlabIdx;
                if (slabIdx != next_sentinel) {
                    pSlab = m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
                }
            }
        }

#if CAPTURE_MEMORY_STATS_SLOW
        *m_pNumItems -= numItemsFreed;
        *m_pNumSlabs -= numSlabsFreed;
#endif

        return numItemsFreed;
    } else {
        uint32_t numItemsBefore = 0;
#if CAPTURE_MEMORY_STATS_SLOW
        cudaMemcpyAsync(&numItemsBefore, m_pNumItems, sizeof(numItemsBefore), cudaMemcpyDeviceToHost);
#endif

        const uint32_t numWorkGroups = std::min((numBuckets() + warpsPerWorkGroup - 1) / warpsPerWorkGroup, maxWorkGroups);
        freeInactiveItems_kernel<<<numWorkGroups, warpsPerWorkGroup * 32>>>(*this, numBuckets());
        CUDA_CHECK_ERROR();

        uint32_t numItemsAfter = 0;
#if CAPTURE_MEMORY_STATS_SLOW
        cudaMemcpy(&numItemsAfter, m_pNumItems, sizeof(numItemsAfter), cudaMemcpyDeviceToHost);
#endif

        checkAlways(numItemsBefore >= numItemsAfter);
        return numItemsBefore - numItemsAfter;
    }
}

#define EXPLICIT_INSTANTIATE_TABLE_BASE(Table)                       \
    template class GpuHashTableBase<Table, EMemoryType::GPU_Malloc>; \
    template class GpuHashTableBase<Table, EMemoryType::GPU_Async>;  \
    template class GpuHashTableBase<Table, EMemoryType::CPU>;

EXPLICIT_INSTANTIATE_TABLE_BASE(Atomic64HashTable)
EXPLICIT_INSTANTIATE_TABLE_BASE(AccelerationHashTable)
EXPLICIT_INSTANTIATE_TABLE_BASE(CompactAccelerationHashTable)
EXPLICIT_INSTANTIATE_TABLE_BASE(TicketBoardHashTable)
