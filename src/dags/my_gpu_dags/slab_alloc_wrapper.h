#pragma once
#include "my_units.h"
#include "slab_alloc.h"
#ifdef __CUDACC__
#include "safe_cooperative_groups.h"
#include <cooperative_groups/reduce.h>
#endif // __CUDACC__
#include "binary_reader.h"
#include "binary_writer.h"
#include <cuda.h>

template <EMemoryType memoryType>
class SlabAllocWrapper {
public:
    HOST static SlabAllocWrapper create(uint32_t numItems, uint32_t itemSizeInU32)
    {
        SlabAllocWrapper out {};
        out.slabAllocLight = ParentAllocator(memoryType);
        checkAlways(itemSizeInU32 <= decltype(slabAllocLight.slab_alloc_context_)::MEM_UNIT_SIZE_);
        return out;
    }
    HOST void release()
    {
        slabAllocLight.free();
    }

    template <EMemoryType newMemoryType>
    SlabAllocWrapper<newMemoryType> copy() const
    {
        SlabAllocWrapper<newMemoryType> out {};
        out.slabAllocLight = ParentAllocator(newMemoryType);
        const size_t memorySize = slabAllocLight.slab_alloc_context_.SUPER_BLOCK_SIZE_ * slabAllocLight.slab_alloc_context_.num_super_blocks_ * sizeof(uint32_t);
        cudaMemcpy(slabAllocLight.d_super_blocks_, out.slabAllocLight.d_super_blocks_, memorySize, cudaMemcpyDefault);
        return out;
    }
    HOST void reserveIfNecessary(size_t additionalSpace)
    {
        checkAlways(false);
    }

    DEVICE void initAsWarp(uint32_t) { }
#ifdef __CUDACC__
    DEVICE uint32_t allocateAsWarp()
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        const auto threadRank = warp.thread_rank();
        const auto warpIdx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

        auto context = *slabAllocLight.getContextPtr();
        context.initAllocator(warpIdx, threadRank);
        return context.warpAllocate(threadRank);
    }
#endif

    HOST_DEVICE uint32_t allocate()
    {
        checkAlways(false);
        return 0;
    }
    HOST_DEVICE void free(uint32_t ptr)
    {
#ifdef __CUDA_ARCH__
        slabAllocLight.getContextPtr()->freeUntouched(ptr);
#else
        checkAlways(false);
#endif
    }

    HOST_DEVICE bool isValidPointer(uint32_t itemIdx) const
    {
        return true;
    }
    HOST_DEVICE uint32_t* decodePointer(uint32_t itemIdx)
    {
        return slabAllocLight.getContextPtr()->getPointerFromSlab(itemIdx, 0);
    }
    HOST_DEVICE const uint32_t* decodePointer(uint32_t itemIdx) const
    {
        return slabAllocLight.getContextPtr()->getPointerFromSlab(itemIdx, 0);
    }
    HOST_DEVICE uint32_t* decodePointer(uint32_t itemIdx, const char*, int)
    {
        return decodePointer(itemIdx);
    }
    HOST_DEVICE const uint32_t* decodePointer(uint32_t itemIdx, const char*, int) const
    {
        return decodePointer(itemIdx);
    }

    HOST size_t size_in_bytes() const
    {
        return 0;
    }
    HOST size_t used_bytes() const
    {
        return 0;
    }
    HOST uint32_t numItemsUsed() const
    {
        return 0;
    }

    HOST my_units::bytes memory_allocated() const
    {
        return 0;
    }
    HOST my_units::bytes memory_used() const
    {
        return 0;
    }
    HOST void writeTo(BinaryWriter& writer) const
    {
    }
    HOST void readFrom(BinaryReader& reader)
    {
    }

    uint32_t itemSizeInU32;

private:
    template <EMemoryType>
    friend class SlabAllocWrapper;

    using ParentAllocator = my_slab_alloc::SlabAllocLight<8, 22, 1>;
    ParentAllocator slabAllocLight;
};