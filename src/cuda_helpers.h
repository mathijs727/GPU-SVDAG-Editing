#pragma once
#include "cuda_error_check.h"
#include "cuda_helpers_cpp.h"
#include "memory.h"
#include <cuda.h>
#include <gsl/gsl-lite.hpp>

#if ENABLE_CHECKS
#define TRACK_GPU_HEAP_MEMORY_USAGE 1
#endif

HOST_DEVICE uint32_t atomicCAS_wrapper(uint32_t* address, uint32_t compare, uint32_t val)
{
#ifdef __CUDA_ARCH__
    return atomicCAS(address, compare, val);
#else
    uint32_t oldVal = *address;
    if (oldVal == compare)
        *address = val;
    return oldVal;
#endif
}
HOST_DEVICE uint64_t atomicCAS_wrapper(uint64_t* address, uint64_t compare, uint64_t val)
{
#ifdef __CUDA_ARCH__
    return atomicCAS((unsigned long long*)address, compare, val);
#else
    uint64_t oldVal = *address;
    if (oldVal == compare)
        *address = val;
    return oldVal;
#endif
}

static HOST_DEVICE uint32_t computeNumWarps(size_t domain)
{
    check(domain > 0);
    return (uint32_t)((domain - 1) / 32 + 1llu);
}

static constexpr uint32_t workGroupSize = 64;
static HOST_DEVICE uint32_t computeNumWorkGroups(size_t domain)
{
    check(domain > 0);
    return (uint32_t)((domain - 1) / workGroupSize + 1llu);
}

#ifdef __CUDACC__

template <typename T>
static __global__ void async_sequence_kernel(gsl::span<T> data, T start = 0, T step = 1)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < data.size())
        data[i] = start + i * step;
}
template <typename T>
void async_sequence(gsl::span<T> data, T start = 0, T step = 1, cudaStream_t stream = nullptr)
{
    if (data.empty())
        return;
    async_sequence_kernel<<<computeNumWorkGroups(data.size()), workGroupSize, 0, stream>>>(data, start, step);
    CUDA_CHECK_ERROR();
}

template <typename T>
__global__ void conditional_scatter_gpu(gsl::span<const T> inItems, gsl::span<const uint32_t> inCondition, gsl::span<const uint32_t> inNewLocation, gsl::span<T> outItems)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inItems.size() && inCondition[i])
        outItems[inNewLocation[i]] = inItems[i];
}
#endif // __CUDACC__

HOST_DEVICE size_t roundUp4(size_t sizeInU32)
{
    return ((sizeInU32 - 1) / 4 + 1) * 4;
}

class WarpComparator {
public:
    WarpComparator(uint32_t sizeInU32)
        : m_sizeInU32(sizeInU32)
        , m_mask(0xFFFFFFFF >> (32 - sizeInU32))
    {
        check(sizeInU32 <= 32);
    }

#ifdef __CUDACC__
    DEVICE bool compare(const uint32_t* lhs, const uint32_t* rhs, uint32_t threadRank) const
    {
        check(__activemask() == 0xFFFFFFFF);
        uint32_t vLhs, vRhs;
        if (threadRank < m_sizeInU32) {
            vLhs = lhs[threadRank];
            vRhs = rhs[threadRank];
        }
        return (__ballot_sync(0xFFFFFFFF, vLhs == vRhs) & m_mask) == m_mask;
    }
    DEVICE bool compare(const uint32_t* lhs, uint32_t rhs, uint32_t threadRank) const
    {
        check(__activemask() == 0xFFFFFFFF);
        // const uint32_t vLhs = threadRank < m_sizeInU32 ? lhs[threadRank] : 0;
        const uint32_t vLhs = lhs[threadRank] * (threadRank < m_sizeInU32);
        return (__ballot_sync(0xFFFFFFFF, vLhs == rhs) & m_mask) == m_mask;
    }
#endif

private:
    uint32_t m_sizeInU32;
    uint32_t m_mask;
};
