#pragma once
#include "array.h"
#include "binary_reader.h"
#include "binary_writer.h"
#include "configuration/gpu_hash_dag_definitions.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/dynamic_slab_alloc.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/individual_chaining_hash_table.h"
#include "dags/my_gpu_dags/slab_alloc_wrapper.h"
#include "memory.h"
#include "my_units.h"
#include "typedefs.h"
#include "utils.h"
#include <concepts>
#include <cstdint>
#include <tuple>
#ifdef __CUDACC__
#include <cooperative_groups.h>
#endif

/* // clang-format off
template <typename T>
concept supports_gpu_base_hash_table = requires(T) {
    { T::next_pointer_offset_in_slab } -> std::same_as<uint32_t&>;
    { T::active_mask_offset_in_slab } -> std::same_as<uint32_t&>;
    { T::slab_init_value } -> std::same_as<uint32_t&>;
    { T::slab_init_size  } -> std::same_as<uint32_t&>;
};
// clang-format on */

#if HASH_TABLE_STORE_SLABS_IN_TABLE
#define INIT_ADD_VARIABLES()                                    \
    const uint32_t offsetInTable = bucketIdx * m_slabSizeInU32; \
    uint32_t slabIdx = base_node_pointer;                       \
    uint32_t* pSlab = &m_table[offsetInTable];
#else
#define INIT_ADD_VARIABLES()               \
    uint32_t slabIdx = m_table[bucketIdx]; \
    uint32_t* pSlab = slabIdx != next_sentinel ? m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__) : nullptr;
#endif

#if HASH_TABLE_STORE_SLABS_IN_TABLE
#define INIT_FIND_VARIABLES()             \
    uint32_t slabIdx = base_node_pointer; \
    const uint32_t* pSlab = &m_table[bucketIdx * m_slabSizeInU32];
#else
#define INIT_FIND_VARIABLES()              \
    uint32_t slabIdx = m_table[bucketIdx]; \
    uint32_t const* pSlab = slabIdx != next_sentinel ? m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__) : nullptr;
#endif

template <template <EMemoryType> typename TChildBase, EMemoryType MemoryType>
class GpuHashTableBase {
private:
    using TChild = TChildBase<MemoryType>;
    // static_assert(supports_gpu_base_hash_table<TChild>);

public:
    static constexpr uint32_t not_found = 0xFFFFFFFF;
    uint32_t itemSizeInU32;

public:
    GpuHashTableBase() = default;
    GpuHashTableBase(uint32_t numBuckets, uint32_t numReservedSlabs, uint32_t slabSizeInU32, uint32_t itemSizeInU32);
    void free();

    template <EMemoryType NewMemoryType>
    TChildBase<NewMemoryType> copy() const
    {
        TChildBase<NewMemoryType> out {};
        out.itemSizeInU32 = itemSizeInU32;
        out.m_slabSizeInU32 = m_slabSizeInU32;
        out.m_numBuckets = m_numBuckets;
        out.m_table = m_table.copy(NewMemoryType);
        out.m_slabAllocator = m_slabAllocator.template copy<NewMemoryType>();
#if CAPTURE_MEMORY_STATS_SLOW
        out.m_pNumItems = Memory::malloc<uint32_t>("GpuHashTableBase::m_pNumItems", sizeof(uint32_t), NewMemoryType);
        out.m_pNumSlabs = Memory::malloc<uint32_t>("GpuHashTableBase::m_pNumSlabs ", sizeof(uint32_t), NewMemoryType);
        if constexpr (MemoryType == EMemoryType::CPU && NewMemoryType == EMemoryType::CPU) {
            *out.m_pNumItems = *m_pNumItems;
            *out.m_pNumSlabs = *m_pNumSlabs;
        } else {
            cudaMemcpy(out.m_pNumItems, m_pNumItems, sizeof(uint32_t), cudaMemcpyDefault);
            cudaMemcpy(out.m_pNumSlabs, m_pNumSlabs, sizeof(uint32_t), cudaMemcpyDefault);
        }
#endif
        return out;
    }

    void reserveIfNecessary(uint32_t numAdditionalItems);
    HOST_DEVICE uint32_t numBuckets() const { return m_numBuckets; }
    HOST_DEVICE bool is_valid() const { return m_table.is_valid(); }

    double currentLoadFactor() const;

    my_units::bytes memory_allocated() const;
    my_units::bytes memory_used_by_items() const;
    my_units::bytes memory_used_by_slabs() const;

    void readFrom(BinaryReader& reader);
    void writeTo(BinaryWriter& writer) const;

    void clearActiveFlags();
    HOST_DEVICE void markAsActive(uint32_t ptr)
    {
        auto [pSlab, itemIdx] = this->decodePointerImpl(ptr);
#ifdef __CUDA_ARCH__
        atomicOr(&pSlab[TChild::active_mask_offset_in_slab], 1u << itemIdx);
#else
        pSlab[TChild::active_mask_offset_in_slab] |= (1u << itemIdx);
#endif
    }
    uint32_t freeInactiveItems();

    HOST_DEVICE static constexpr uint32_t encodePointer(uint32_t bucketIdx, uint32_t slabIdx, uint32_t itemIdx)
    {
        if (slabIdx == base_node_pointer) {
            check((bucketIdx & slab_mask) == bucketIdx);
            check((itemIdx & item_mask) == itemIdx);
            return (bucketIdx << slab_offset_in_pointer) | (1 << in_base_node_offset_in_pointer) | (itemIdx << item_offset_in_pointer);
        } else {
            check((slabIdx & slab_mask) == slabIdx);
            check((itemIdx & item_mask) == itemIdx);
            return (slabIdx << slab_offset_in_pointer) | (0 << in_base_node_offset_in_pointer) | (itemIdx << item_offset_in_pointer);
        }
    }

protected:
    HOST_DEVICE void allocateNewSlabThread(uint32_t bucketIdx, uint32_t& outSlabIdx, uint32_t*& pOutSlab)
    {
        outSlabIdx = m_slabAllocator.allocate();
        pOutSlab = m_slabAllocator.decodePointer(outSlabIdx, __FILE__, __LINE__);
        for (uint32_t i = 0; i < TChild::slab_init_size; ++i)
            pOutSlab[i] = TChild::slab_init_value;
        check(((uintptr_t)pOutSlab % 128) == 0);
        //printf("alignment: %u\n", (unsigned)((uintptr_t)pOutSlab % 128));

#if HASH_TABLE_STORE_SLABS_IN_TABLE
        uint32_t* const pBaseSlabNextPointer = &m_table[(bucketIdx * m_slabSizeInU32) + TChild::next_pointer_offset_in_slab];
#else
        uint32_t* const pBaseSlabNextPointer = &m_table[bucketIdx];
#endif
        pOutSlab[TChild::next_pointer_offset_in_slab] = *pBaseSlabNextPointer;

#ifdef __CUDA_ARCH__

        __threadfence();
        const uint32_t prevHead = atomicCAS(pBaseSlabNextPointer, pOutSlab[TChild::next_pointer_offset_in_slab], outSlabIdx);
        if (prevHead != pOutSlab[TChild::next_pointer_offset_in_slab]) {
            m_slabAllocator.free(outSlabIdx);
            outSlabIdx = prevHead;
            pOutSlab = m_slabAllocator.decodePointer(outSlabIdx, __FILE__, __LINE__);
        }
#if CAPTURE_MEMORY_STATS_SLOW
        else {
            atomicAdd(m_pNumSlabs, 1);
        }
#endif // CAPTURE_MEMORY_STATS_SLOW

        // Don't remove this; will crash otherwise. Don't know why.
        __threadfence();

#else // __CUDA_ARCH__

        *pBaseSlabNextPointer = outSlabIdx;
#if CAPTURE_MEMORY_STATS_SLOW
        *m_pNumSlabs += 1;
#endif // CAPTURE_MEMORY_STATS_SLOW

#endif // __CUDA_ARCH__
    }

    DEVICE void allocateNewSlabWarp(uint32_t inThreadRank, uint32_t bucketIdx, uint32_t& outSlabIdx, uint32_t*& pOutSlab)
    {
#ifdef __CUDA_ARCH__
        // Allocate the slab.
        outSlabIdx = m_slabAllocator.allocateAsWarp();
        pOutSlab = m_slabAllocator.decodePointer(outSlabIdx, __FILE__, __LINE__);

        // Initialize the memory.
        static_assert(TChild::slab_init_size == 32 || TChild::slab_init_size == 64);
        check(((uintptr_t)pOutSlab % 128) == 0);
        //printf("alignment: %u\n", (unsigned)((uintptr_t)pOutSlab % 128));
        pOutSlab[inThreadRank] = TChild::slab_init_value;
        if constexpr (TChild::slab_init_size > 32) {
            pOutSlab[32 + inThreadRank] = TChild::slab_init_value;
        }
        // We might have written to the next pointer memory location.
        // This will be overwritten but we need to make sure that this happens in the correct order.
        __syncwarp();

#if HASH_TABLE_STORE_SLABS_IN_TABLE
        uint32_t* const pBaseSlabNextPointer = &m_table[(bucketIdx * m_slabSizeInU32) + TChild::next_pointer_offset_in_slab];
#else
        uint32_t* const pBaseSlabNextPointer = &m_table[bucketIdx];
#endif
        // uint32_t* const pBaseSlabNextPointer = &m_table[inSlabStartInTable];
        static constexpr uint32_t next_pointer_thread = TChild::next_pointer_offset_in_slab % 32;
        static_assert(next_pointer_thread < 32);
        if (inThreadRank == next_pointer_thread) {
            pOutSlab[TChild::next_pointer_offset_in_slab] = *pBaseSlabNextPointer;
            __threadfence(); // Make sure write is visible to other threads.
            const uint32_t prevHead = atomicCAS(pBaseSlabNextPointer, pOutSlab[TChild::next_pointer_offset_in_slab], outSlabIdx);
            if (prevHead != pOutSlab[TChild::next_pointer_offset_in_slab]) {
                m_slabAllocator.free(outSlabIdx);
                outSlabIdx = prevHead;
                pOutSlab = m_slabAllocator.decodePointer(outSlabIdx, __FILE__, __LINE__);
            }
#if CAPTURE_MEMORY_STATS_SLOW
            else {
                atomicAdd(m_pNumSlabs, 1);
            }
#endif // CAPTURE_MEMORY_STATS_SLOW
        }

        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        outSlabIdx = warp.shfl(outSlabIdx, next_pointer_thread);
        pOutSlab = warp.shfl(pOutSlab, next_pointer_thread);
#endif // __CUDA_ARCH__
    }

    HOST_DEVICE void counterItemAdded()
    {
#if CAPTURE_MEMORY_STATS_SLOW
#ifdef __CUDA_ARCH__
        atomicAdd(m_pNumItems, 1);
#else
        *m_pNumItems += 1;
#endif
#endif
    }
    DEVICE void counterItemAddedAsWarp(uint32_t threadRank)
    {
#if CAPTURE_MEMORY_STATS_SLOW && defined(__CUDA_ARCH__)
        if (threadRank == 0)
            atomicAdd(m_pNumItems, 1);
#endif
    }

#ifdef __CUDACC__
    DEVICE uint32_t computeBucketAsWarp(uint32_t const* pItem, uint32_t item, uint32_t threadRank) const
    {
        uint32_t hash = 0;
        if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::Murmur) {
            hash = Utils::murmurhash32xN(pItem, itemSizeInU32);
        } else if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::MurmurXor) {
            hash = Utils::xorMurmurHash32xN_warp(item, itemSizeInU32, threadRank);
        } else if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::SlabHashXor) {
            hash = Utils::xorHash32xN_warp(item, itemSizeInU32, threadRank);
        } else if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::SlabHashBoostCombine) {
            hash = Utils::boostCombineHash32xN_warp(item, itemSizeInU32, threadRank);
        } else if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::SlabHashSingle) {
            hash = Utils::slabHash(pItem[0]);
        }
        return hash % m_numBuckets;
    }
#endif

    HOST_DEVICE uint32_t computeBucket(uint32_t const* pItem) const
    {
        uint32_t hash = 0;
        if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::Murmur) {
            hash = Utils::murmurhash32xN(pItem, itemSizeInU32);
        } else if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::MurmurXor) {
            hash = Utils::xorMurmurHash32xN(pItem, itemSizeInU32);
        } else if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::SlabHashXor) {
            hash = Utils::xorHash32xN(pItem, itemSizeInU32);
        } else if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::SlabHashBoostCombine) {
            hash = Utils::boostCombineHash32xN(pItem, itemSizeInU32);
        } else if constexpr (HASH_TABLE_HASH_METHOD == HashMethod::SlabHashSingle) {
            hash = Utils::slabHash(pItem[0]);
        }
        return hash % m_numBuckets;
    }

    HOST_DEVICE std::pair<const uint32_t*, uint32_t> decodePointerImpl(uint32_t ptr) const
    {
        check(m_slabAllocator.isValidPointer(ptr >> item_bits));
        const uint32_t slabIdx = (ptr >> slab_offset_in_pointer) & slab_mask;
        const uint32_t inBaseNode = (ptr >> in_base_node_offset_in_pointer) & in_base_node_bits;
        const uint32_t itemIdx = (ptr >> item_offset_in_pointer) & item_mask;

        const uint32_t* pSlab = inBaseNode ? &m_table[slabIdx * m_slabSizeInU32] : m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
        return { pSlab, itemIdx };
    }
    HOST_DEVICE std::pair<uint32_t*, uint32_t> decodePointerImpl(uint32_t ptr)
    {
        check(m_slabAllocator.isValidPointer(ptr >> item_bits));
        const uint32_t slabIdx = (ptr >> slab_offset_in_pointer) & slab_mask;
        const uint32_t inBaseNode = (ptr >> in_base_node_offset_in_pointer) & in_base_node_bits;
        const uint32_t itemIdx = (ptr >> item_offset_in_pointer) & item_mask;

        uint32_t* pSlab = inBaseNode ? &m_table[slabIdx * m_slabSizeInU32] : m_slabAllocator.decodePointer(slabIdx, __FILE__, __LINE__);
        return { pSlab, itemIdx };
    }

public: // Need to be public so they can be accessed from within a CUDA kernel.
    __host__ __device__ void _clearActiveFlagsBucket(uint32_t bucketIdx);
    __device__ void _freeInactiveItemsBucket_warp(uint32_t bucketIdx);

protected:
    // Pointer encoding.
    constexpr static uint32_t item_bits = 5;
    constexpr static uint32_t in_base_node_bits = 1;
    constexpr static uint32_t slab_bits = 32 - item_bits - in_base_node_bits;
    constexpr static uint32_t slab_mask = (1u << slab_bits) - 1u;
    constexpr static uint32_t in_base_node_mask = (1u << in_base_node_bits) - 1u;
    constexpr static uint32_t item_mask = (1u << item_bits) - 1u;
    constexpr static uint32_t item_offset_in_pointer = 0;
    constexpr static uint32_t in_base_node_offset_in_pointer = item_offset_in_pointer + item_bits;
    constexpr static uint32_t slab_offset_in_pointer = in_base_node_offset_in_pointer + in_base_node_bits;

    constexpr static uint32_t next_sentinel = 0xFFFFFFFF;
    constexpr static uint32_t base_node_pointer = 0xFFFFFFFE;

protected:
#if CAPTURE_MEMORY_STATS_SLOW
    uint32_t *m_pNumItems, *m_pNumSlabs;
#endif

    uint32_t m_slabSizeInU32;
    uint32_t m_numBuckets;

    StaticArray<uint32_t, uint32_t, 128> m_table;
    DynamicSlabAllocator<128, MemoryType> m_slabAllocator;
    // SlabAllocWrapper<MemoryType> m_slabAllocator;
};
