#pragma once
#include "array.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "memory.h"
#include <array>
#include <cuda.h>

class GPUTimingsManager;

template <uint32_t MaxSizeInU32_>
struct VariableSizeItem {
    static constexpr uint32_t MaxSizeInU32 = MaxSizeInU32_;
    // uint32_t padding[MaxSizeInU32_];
    std::array<uint32_t, MaxSizeInU32_> padding;
    uint32_t size = 0;

    constexpr bool operator==(const VariableSizeItem<MaxSizeInU32_>& rhs) const noexcept
    {
        check(size > 0);
        check(size <= MaxSizeInU32);
        if (rhs.size != size)
            return false;

        for (uint32_t i = 0; i < size; ++i) {
            if (padding[i] != rhs.padding[i])
                return false;
        }
        return true;
    }
    constexpr bool operator!=(const VariableSizeItem<MaxSizeInU32_>& rhs) const noexcept
    {
        return !(*this == rhs);
    }
    constexpr bool operator<(const VariableSizeItem<MaxSizeInU32_>& rhs) const noexcept
    {
        check(size > 0);
        check(size <= MaxSizeInU32);
        if (size < rhs.size)
            return true;
        else if (size > rhs.size)
            return false;

        for (uint32_t i = 0; i < size; ++i) {
            if (padding[i] < rhs.padding[i])
                return true;
            else if (padding[i] > rhs.padding[i])
                return false;
            // else: check next uint32_t...
        }
        return false;
    }
};

static constexpr uint32_t sentinel = (uint32_t)-1;
struct IntermediateSVO {
public:
    static constexpr uint32_t nodeSizeInU32 = MyGPUHashDAG<EMemoryType::GPU_Malloc>::maxNodeSizeInU32;
    static constexpr uint32_t leafSizeInU32 = MyGPUHashDAG<EMemoryType::GPU_Malloc>::maxLeafSizeInU32;

    using Node = VariableSizeItem<nodeSizeInU32>;
    using Leaf = VariableSizeItem<leafSizeInU32>;

    std::array<gsl::span<Node>, MAX_LEVELS> innerNodes;
    gsl::span<Leaf> leaves;

public:
    void free(GpuMemoryPool& memPool);
    size_t size_in_bytes() const;
    size_t largestLevel() const;
};

template <typename DAG, typename Editor>
IntermediateSVO create_edit_intermediate_svo(
    const DAG& inDag, const Editor& editor,
    GpuMemoryPool& memPool, cudaStream_t stream, GPUTimingsManager& timingsManager);
