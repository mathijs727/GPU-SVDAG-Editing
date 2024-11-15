#pragma once
#include "memory.h"
#include <cstddef>
#include <cuda.h>
#include <gsl/gsl-lite.hpp>

// This header file contains helper functions which can be included from regular *.cpp files (not requiring nvcc compiler).

void deviceMemset32Async(uint32_t* dataU32, uint32_t value, size_t sizeInU32, cudaStream_t stream = 0);
void cudaDeviceSynchronizeOnWindows();

template <typename F>
void call_cub(F&& f, cudaStream_t stream)
{
    // Calling cub with nullptr will set tmpStorageSize (passed as reference) to the memory required in bytes.
    size_t tmpStorageSize = 0;
    f(nullptr, tmpStorageSize);

    void* memory;
    CUDA_CHECKED_CALL cudaMallocAsync(&memory, tmpStorageSize, stream);
    f(memory, tmpStorageSize);
    CUDA_CHECKED_CALL cudaFreeAsync(memory, stream);
}

template <typename T>
gsl::span<T> mallocRange(const char* name, size_t itemCount, EMemoryType memoryType)
{
    T* pMemory = Memory::malloc<T>(name, itemCount * sizeof(T), memoryType);
    return gsl::span(pMemory, itemCount);
}
template <typename T>
gsl::span<T> cudaMallocRange(size_t itemCount)
{
    CUDA_CHECK_ERROR();
    T* pMemory;
    CUDA_CHECKED_CALL cudaMalloc(&pMemory, itemCount * sizeof(T));
    CUDA_CHECK_ERROR();
    return gsl::span(pMemory, itemCount);
}
template <typename T>
gsl::span<T> cudaMallocAsyncRange(size_t itemCount, cudaStream_t stream)
{
    CUDA_CHECK_ERROR();
    T* pMemory;
    CUDA_CHECKED_CALL cudaMallocAsync(&pMemory, itemCount * sizeof(T), stream);
    CUDA_CHECK_ERROR();
    return gsl::span(pMemory, itemCount);
}
template <typename T>
gsl::span<T> cudaMallocAsyncRange(size_t itemCount, cudaMemPool_t memPool, cudaStream_t stream)
{
    CUDA_CHECK_ERROR();
    T* pMemory;
    cudaMallocFromPoolAsync(&pMemory, itemCount * sizeof(T), memPool, stream);
    CUDA_CHECK_ERROR();
    return gsl::span(pMemory, itemCount);
}
template <typename T>
gsl::span<T> cudaMallocPinnedRange(size_t itemCount)
{
    CUDA_CHECK_ERROR();
    T* pMemory;
    cudaMallocHost(&pMemory, itemCount * sizeof(T));
    CUDA_CHECK_ERROR();
    return gsl::span(pMemory, itemCount);
}
template <typename T>
void cudaFreePinnedRange(gsl::span<T> items)
{
    CUDA_CHECK_ERROR();
    cudaFreeHost(items.data());
    CUDA_CHECK_ERROR();
}
