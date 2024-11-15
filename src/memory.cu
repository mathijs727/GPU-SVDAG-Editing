// #undef ENABLE_VULKAN_MEMORY_ALLOCATOR
#define VMA_IMPLEMENTATION 1
#include "cuda_error_check.h"
#include "memory.h"
#include "typedefs.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <new>
#include <nvml.h>
#include <sstream>
#include <vector>
#ifdef _WIN32
#include <malloc.h>
#endif

#if TRACK_GLOBAL_NEWDELETE

struct ControlBlock {
    void* pBase;
    std::size_t blockSize;
};

// To overload all invocations of operator new, we only need to overload the aligned & non-aligned throwing single-object operator.
// All other versions implemented by the standard library will call these two functions.
//
// https://en.cppreference.com/w/cpp/memory/new/operator_new
// The standard library implementations of the nothrow versions (5-8) directly calls the corresponding throwing versions (1-4).
// The standard library implementation of the throwing array versions (2,4) directly calls the corresponding single-object version (1,3).
// Thus, replacing the throwing single object allocation functions is sufficient to handle all allocations.
void* operator new(std::size_t count)
{
    const size_t allocationSize = sizeof(ControlBlock) + count;
    Memory::track_add_memory(allocationSize);
    if (auto* ptr = (std::byte*)std::malloc(allocationSize)) {
        auto* cb = reinterpret_cast<ControlBlock*>(ptr);
        cb->pBase = ptr;
        cb->blockSize = allocationSize;
        return ptr + sizeof(ControlBlock);
    }
    throw std::bad_alloc();
}

void* operator new(std::size_t count, std::align_val_t alignmentT)
{
    const size_t alignment = static_cast<size_t>(alignmentT);
    if (alignment < std::alignment_of_v<ControlBlock>)
        throw std::bad_alloc();

    const size_t allocationSize = sizeof(ControlBlock) + count + alignment;
    Memory::track_add_memory(allocationSize);
    if (auto* basePtr = (std::byte*)std::malloc(allocationSize)) {
        std::byte* ptr = basePtr + sizeof(ControlBlock);
        size_t freeSpace = allocationSize - sizeof(ControlBlock);
        if (std::align(alignment, count, reinterpret_cast<void*&>(ptr), freeSpace)) {
            auto* cb = reinterpret_cast<ControlBlock*>(ptr - sizeof(ControlBlock));
            cb->pBase = basePtr;
            cb->blockSize = allocationSize;
            return ptr;
        }
    }
    throw std::bad_alloc();
}

// Similar to the new operator we only have to overload the throwing single-object version to cover all cases
//
// https://en.cppreference.com/w/cpp/memory/new/operator_delete
// The standard library implementations of the nothrow versions (9,10) directly call the corresponding throwing versions (1,2).
// The standard library implementations of the size-aware deallocation functions (5-8) directly call the corresponding size-unaware deallocation functions (1-4).
// The standard library implementations of size-unaware throwing array forms (2,4) directly calls the corresponding single-object forms (1,3).
// Thus, replacing the throwing single object deallocation functions(1, 3) is sufficient to handle all deallocations.
void operator delete(void* ptr)
{
    if (ptr) {
        auto* cb = reinterpret_cast<ControlBlock*>(reinterpret_cast<std::byte*>(ptr) - sizeof(ControlBlock));
        Memory::track_del_memory(cb->blockSize);
        std::free(cb->pBase);
    }
}

void operator delete(void* ptr, std::align_val_t)
{
    if (ptr) {
        auto* cb = reinterpret_cast<ControlBlock*>(reinterpret_cast<std::byte*>(ptr) - sizeof(ControlBlock));
        Memory::track_del_memory(cb->blockSize);
        std::free(cb->pBase);
    }
}
#endif // ~ TRACK_GLOBAL_NEWDELETE

Memory Memory::singleton;

inline bool is_gpu_type(EMemoryType type)
{
    return type == EMemoryType::GPU_Malloc || type == EMemoryType::GPU_Async || type == EMemoryType::GPU_Managed;
}

inline std::string type_to_string(EMemoryType type)
{
    switch (type) {
    case EMemoryType::GPU_Managed:
        return "GPU Managed";
    case EMemoryType::GPU_Async:
        return "GPU Malloc (Async)";
    case EMemoryType::GPU_Malloc:
        return "GPU Malloc";
    case EMemoryType::CPU:
        return "CPU Malloc";
    default:
        check(false);
        return "ERROR";
    }
}

Memory::~Memory()
{
    check(this == &singleton);

    for (auto& it : allocations) {
        printf("%s (%p) leaked %fMB\n", it.second.name, it.first, Utils::to_MB(it.second.size));
    }
    checkAlways(allocations.empty());
    if (allocations.empty()) {
        printf("No leaks!\n");
    }
}

void Memory::cuda_memcpy_impl(uint8* dst, const uint8* src, uint64 size, cudaMemcpyKind memcpyKind)
{
    const auto BlockCopy = [&]() {
        const double Start = Utils::seconds();
        CUDA_CHECKED_CALL cudaMemcpy(dst, src, size, memcpyKind);
        const double End = Utils::seconds();

        return (double)size / double(1u << 30) / (End - Start);
    };

    if (memcpyKind == cudaMemcpyDeviceToDevice) {
        PROFILE_SCOPEF("Memcpy HtH %fMB", size / double(1u << 20));
        [[maybe_unused]] const double Bandwidth = BlockCopy();
        ZONE_METADATA("%fGB/s", Bandwidth);
    } else if (memcpyKind == cudaMemcpyDeviceToHost) {
        PROFILE_SCOPEF("Memcpy DtH %fMB", size / double(1u << 20));
        [[maybe_unused]] const double Bandwidth = BlockCopy();
        ZONE_METADATA("%fGB/s", Bandwidth);
    } else if (memcpyKind == cudaMemcpyHostToDevice) {
        PROFILE_SCOPEF("Memcpy HtD %fMB", size / double(1u << 20));
        [[maybe_unused]] const double Bandwidth = BlockCopy();
        ZONE_METADATA("%fGB/s", Bandwidth);
    } else if (memcpyKind == cudaMemcpyDefault) {
        PROFILE_SCOPEF("Memcpy Default %fMB", size / double(1u << 20));
        [[maybe_unused]] const double Bandwidth = BlockCopy();
        ZONE_METADATA("%fGB/s", Bandwidth);
    }
}

void* Memory::malloc_impl(size_t size, size_t alignment, const char* name, EMemoryType type)
{
    checkAlways(size != 0);
    // printf("Allocating %fMB for %s\n", Utils::to_MB(size), name.c_str());
    void* ptr = nullptr;
    if (is_gpu_type(type)) {
        PROFILE_SCOPE("Malloc GPU");
        // printf("cudaMalloc(%zuMB)\n", size>>20);
        cudaError_t error;
        if (type == EMemoryType::GPU_Managed) {
            PROFILE_SCOPE("cudaMallocManaged");
#ifdef ENABLE_VULKAN_MEMORY_ALLOCATOR
            ptr = managedAllocator.malloc(size, alignment, error);
#else // ENABLE_VULKAN_MEMORY_ALLOCATOR
            error = cudaMallocManaged(&ptr, size);
            if (MEM_ADVISE_READ_MOSTLY) {
                // Huge performance improvements when reading
                CUDA_CHECKED_CALL cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, 0);
            }
#endif // ENABLE_VULKAN_MEMORY_ALLOCATOR
        } else if (type == EMemoryType::GPU_Async) {
            PROFILE_SCOPE("cudaMallocAsync");
#ifdef ENABLE_VULKAN_MEMORY_ALLOCATOR
            ptr = gpuAllocator.malloc(size, alignment, error);
#else // ENABLE_VULKAN_MEMORY_ALLOCATOR
            error = cudaMallocAsync(&ptr, size, 0);
#endif // ENABLE_VULKAN_MEMORY_ALLOCATOR
        } else {
            check(type == EMemoryType::GPU_Malloc);
            PROFILE_SCOPE("cudaMalloc");
#ifdef ENABLE_VULKAN_MEMORY_ALLOCATOR
            ptr = gpuAllocator.malloc(size, alignment, error);
#else // ENABLE_VULKAN_MEMORY_ALLOCATOR
      // error = cudaMallocAsync(&ptr, size, 0);
            error = cudaMalloc(&ptr, size);
#endif // ENABLE_VULKAN_MEMORY_ALLOCATOR
        }
        CUDA_CHECK_ERROR();
        if (error != cudaSuccess) {
            printf("\n\n\n");
            printf("Fatal error when allocating %zu bytes of memory!\n", size);
            std::cout << get_stats_string_impl() << std::endl;
        }
        totalUsedGPUMemory += size;
    } else {
        check(type == EMemoryType::CPU);
        PROFILE_SCOPE("Malloc CPU");
        if (alignment == 0) {
            ptr = std::malloc(size);
        } else {
#ifdef _WIN32
            ptr = _aligned_malloc(size, alignment);
            alignedMallocPointers.insert(ptr);
#else
            ptr = std::aligned_alloc(alignment, size);
#endif
        }
        totalAllocatedCPUMemory += size;
    }

    {
        PROFILE_SCOPE("update allocations");
        checkAlways(allocations.find(ptr) == allocations.end());
        allocations[ptr] = { name, type, size };
    }
    TRACE_ALLOC(ptr, size);
    if (alignment != 0)
        checkAlways((uintptr_t)ptr % alignment == 0);

    return ptr;
}

void Memory::free_impl(void* ptr)
{
    if (!ptr)
        return;

    checkAlways(allocations.find(ptr) != allocations.end());
    auto& alloc = allocations[ptr];
    if (is_gpu_type(alloc.type)) {
        PROFILE_SCOPE("Free GPU");
#ifdef ENABLE_VULKAN_MEMORY_ALLOCATOR
        if (!gpuAllocator.free(ptr) && !managedAllocator.free(ptr)) {
            check(false);
        }
#else
        if (alloc.type == EMemoryType::GPU_Managed)
            CUDA_CHECKED_CALL cudaFree(ptr);
        else if (alloc.type == EMemoryType::GPU_Async)
            CUDA_CHECKED_CALL cudaFreeAsync(ptr, nullptr);
        else if (alloc.type == EMemoryType::GPU_Malloc)
            CUDA_CHECKED_CALL cudaFree(ptr);
        else
            checkAlways(false);
#endif
        totalUsedGPUMemory -= alloc.size;
    } else {
        check(alloc.type == EMemoryType::CPU);
        PROFILE_SCOPE("Free CPU");
#ifdef _WIN32
        if (auto iter = alignedMallocPointers.find(ptr); iter != std::end(alignedMallocPointers)) {
            _aligned_free(ptr);
            alignedMallocPointers.erase(iter);
        } else {
            std::free(ptr);
        }
#else
        std::free(ptr);
#endif
        totalAllocatedCPUMemory -= alloc.size;
    }
    check(ptr != nullptr);
    check(allocations.find(ptr) != std::end(allocations));
    allocations.erase(ptr);
    TRACE_FREE(ptr);
}

#ifdef ENABLE_VULKAN_MEMORY_ALLOCATOR
Memory::VmaBlockAllocator::VmaBlockAllocator(EMemoryType memoryType)
    : m_memoryType(memoryType)
{
}

Memory::VmaBlockAllocator::~VmaBlockAllocator()
{
    check(m_allocations.empty());
    for (auto& block : m_blocks) {
        CUDA_CHECKED_CALL cudaFree(block.pMemory);
        vmaDestroyVirtualBlock(block.block);
    }
}

constexpr uint32_t typicalBlockSize = 128llu * 1024llu * 1024llu;

void* Memory::VmaBlockAllocator::malloc(size_t size, size_t alignment, cudaError_t& error)
{
    VmaVirtualAllocationCreateInfo allocCreateInfo {};
    allocCreateInfo.size = size;
    allocCreateInfo.alignment = alignment ? alignment : 8;

    // Try to allocate from a block. Return nullptr if block is full.
    auto tryAllocFromBlock = [&](auto iter) -> void* {
        VmaVirtualAllocation alloc;
        VkDeviceSize offset;
        auto res = vmaVirtualAllocate(iter->block, &allocCreateInfo, &alloc, &offset);
        if (res == VK_SUCCESS) {
            error = cudaSuccess;
            void* pOut = (void*)(iter->pMemory + offset);
            assert(m_allocations.find(pOut) == std::end(m_allocations)); // Check causes compile error on Windows (using asm("trap") in MSVC)
            m_allocations[pOut] = { iter, alloc };
            ++iter->numAllocations;
            return pOut;
        } else {
            return nullptr;
        }
    };

    if (!m_blocks.empty()) {
        /*// Loop over all blocks and try to allocate from it (starting at last block from which we allocated successfully).
        size_t startBlockIdx = m_currentBlockIdx;
        do {
            void* out = tryAllocFromBlock(m_blocks[m_currentBlockIdx]);
            if (out)
                return out;
            m_currentBlockIdx = (m_currentBlockIdx + 1) % m_blocks.size();
        } while (m_currentBlockIdx != startBlockIdx);*/
        for (auto iter = std::begin(m_blocks); iter != std::end(m_blocks); ++iter) {
            void* out = tryAllocFromBlock(iter);
            if (out)
                return out;
        }
    }

    // Could not find a block with enough free space. Allocate a new one.
    const auto blockSize = std::max(typicalBlockSize, (uint32_t)size + 1024u);
    MemoryBlock newBlock;
    {
        PROFILE_SCOPE("cudaMallocManaged/Async");
        CUDA_CHECKED_CALL(m_memoryType == EMemoryType::GPU_Managed ? cudaMallocManaged(&newBlock.pMemory, blockSize) : cudaMallocAsync(&newBlock.pMemory, blockSize, nullptr));
    }
    newBlock.sizeInBytes = blockSize;
    if (!newBlock.pMemory) {
        printf("Failed to allocate %uKB of memory\n", blockSize >> 10);
        printf("Currently memory usage: %zuMB (CPU), %zuMB (GPU)\n", singleton.totalAllocatedCPUMemory >> 20, singleton.totalUsedGPUMemory >> 20);
        const auto statsStr = singleton.get_stats_string_impl();
        printf("%s\n", statsStr.c_str());
    }
    checkAlways(newBlock.pMemory);
    // cudaMemset(newBlock.pMemory, 0, blockSize);
    m_currentBlockIdx = m_blocks.size();
    VmaVirtualBlockCreateInfo blockCreateInfo {};
    blockCreateInfo.size = blockSize;
    checkAlways(vmaCreateVirtualBlock(&blockCreateInfo, &newBlock.block) == VK_SUCCESS);
    m_blocks.emplace_back(std::move(newBlock));

    singleton.totalAllocatedGPUMemory += blockSize;
    if (singleton.totalAllocatedGPUMemory > singleton.peakAllocatedGPUMemory) {
        // printf("Peak GPU memory: %zu MB\n", singleton.totalAllocatedGPUMemory >> 20);
        singleton.peakAllocatedGPUMemory = singleton.totalAllocatedGPUMemory;
    }

    // Allocate from the newly allocated block.
    auto out = tryAllocFromBlock(--std::end(m_blocks));
    checkAlways(out != nullptr);
    return out;
}

bool Memory::VmaBlockAllocator::free(void* ptr)
{
    if (auto iter = m_allocations.find(ptr); iter != std::end(m_allocations)) {
        const auto& [blockIter, vmaAllocation] = iter->second;
        vmaVirtualFree(blockIter->block, vmaAllocation);
        // If block becomes empty then release the memory.
        if (--blockIter->numAllocations == 0) {
            if (m_memoryType == EMemoryType::GPU_Malloc)
                CUDA_CHECKED_CALL cudaFreeAsync(blockIter->pMemory, nullptr);
            else
                CUDA_CHECKED_CALL cudaFree(blockIter->pMemory);
            vmaDestroyVirtualBlock(blockIter->block);
            Memory::singleton.totalAllocatedGPUMemory -= blockIter->sizeInBytes;
            m_blocks.erase(blockIter);
        }
        m_allocations.erase(iter);
        return true;
    }
    return false;
}

Memory::VmaAllocator::VmaAllocator(EMemoryType memoryType, size_t memoryPoolSize)
{
    if (memoryType == EMemoryType::GPU_Managed)
        CUDA_CHECKED_CALL cudaMallocManaged(&m_pData, memoryPoolSize);
    else
        CUDA_CHECKED_CALL cudaMalloc(&m_pData, memoryPoolSize);

    VmaVirtualBlockCreateInfo blockCreateInfo {};
    blockCreateInfo.size = memoryPoolSize;
    checkAlways(vmaCreateVirtualBlock(&blockCreateInfo, &m_vmaBlock) == VK_SUCCESS);
}

Memory::VmaAllocator::~VmaAllocator()
{
    vmaDestroyVirtualBlock(m_vmaBlock);
    CUDA_CHECKED_CALL cudaFree(m_pData);
}

void* Memory::VmaAllocator::malloc(size_t size, size_t alignment, cudaError_t& error)
{
    VmaVirtualAllocationCreateInfo allocCreateInfo {};
    allocCreateInfo.size = size;
    allocCreateInfo.alignment = alignment ? alignment : 8;

    VmaVirtualAllocation alloc;
    VkDeviceSize offset;
    auto res = vmaVirtualAllocate(m_vmaBlock, &allocCreateInfo, &alloc, &offset);
    if (res == VK_SUCCESS) {
        error = cudaSuccess;
        void* pOut = m_pData + offset;
        m_allocations[pOut] = alloc;
        return pOut;
    } else {
        printf("size = %zuKB\n", size >> 10);
        VmaDetailedStatistics stats;
        vmaCalculateVirtualBlockStatistics(m_vmaBlock, &stats);
        printf("My virtual block has %zu bytes used by %u virtual allocations\n", stats.statistics.allocationBytes, stats.statistics.allocationCount);
        checkAlways(false);
        return nullptr;
    }
}

bool Memory::VmaAllocator::free(void* pointer)
{
    if (auto iter = m_allocations.find(pointer); iter != std::end(m_allocations)) {
        vmaVirtualFree(m_vmaBlock, iter->second);
        m_allocations.erase(iter);
        return true;
    } else {
        return false;
    }
}

#endif

void Memory::realloc_impl(void*& ptr, size_t newSize, size_t newAlignment)
{
    checkAlways(ptr);
    checkAlways(allocations.find(ptr) != allocations.end());
    const auto oldPtr = ptr;
    const auto oldAlloc = allocations[ptr];

    // printf("reallocating %s (%s)\n", oldAlloc.name, type_to_string(oldAlloc.type).c_str());

    if (is_gpu_type(oldAlloc.type)) {
        ptr = malloc_impl(newSize, newAlignment, oldAlloc.name, oldAlloc.type);
        if (ptr) {
            if (oldAlloc.type == EMemoryType::GPU_Async) {
                CUDA_CHECKED_CALL cudaMemcpyAsync(ptr, oldPtr, oldAlloc.size, cudaMemcpyDefault);
            } else {
                cuda_memcpy_impl(static_cast<uint8*>(ptr), static_cast<uint8*>(oldPtr), oldAlloc.size, cudaMemcpyDefault);
            }
        }
        free_impl(oldPtr);
    } else {
        check(oldAlloc.type == EMemoryType::CPU);
        allocations.erase(ptr);
        TRACE_FREE(ptr);
        ptr = std::realloc(ptr, newSize);
        TRACE_ALLOC(ptr, newSize);
        allocations[ptr] = { oldAlloc.name, oldAlloc.type, oldAlloc.size };
    }

    if (newAlignment != 0)
        checkAlways((uintptr_t)ptr % newAlignment == 0);
}

const char* Memory::get_alloc_name_impl(void* ptr) const
{
    checkAlways(ptr);
    checkAlways(allocations.find(ptr) != allocations.end());
    return allocations.at(ptr).name;
}

std::string Memory::get_stats_string_impl() const
{
    struct AllocCount {
        size_t size = 0;
        size_t count = 0;
    };
    std::unordered_map<std::string, AllocCount> map;
    for (auto& it : allocations) {
        auto name = std::string(it.second.name) + " (" + type_to_string(it.second.type) + ")";
        map[name].size += it.second.size;
        map[name].count++;
    }

    std::vector<std::pair<std::string, AllocCount>> list(map.begin(), map.end());
    std::sort(list.begin(), list.end(), [](auto& a, auto& b) { return a.second.size > b.second.size; });

    std::stringstream ss;
    ss << "GPU memory: " << Utils::to_MB(totalUsedGPUMemory) << "MB" << std::endl;
    ss << "CPU memory: " << Utils::to_MB(totalAllocatedCPUMemory) << "MB" << std::endl;
    for (auto& it : list) {
        ss << it.first << ": " << Utils::to_MB(it.second.size) << "MB (" << it.second.count << " allocs)" << std::endl;
    }
    return ss.str();
}

size_t Memory::availableDeviceMemory() const
{
    // On Windows, the amount of free memory returned by `cudaMemGetInfo` is (much) higher than the actual amount of free memory.
    // This leads to overallocation and subsequently a significant slow-down in performance.
    // size_t free, total;
    // cudaMemGetInfo(&free, &total);
    // return free;

    const auto checkError = [](nvmlReturn_t status) {
        if (status != NVML_SUCCESS)
            printf("NVML ERROR: %s\n", nvmlErrorString(status));
    };

    static nvmlReturn_t tmp = []() { return nvmlInit(); }(); // Call nvmlInit only once

    nvmlDevice_t device;
    checkError(nvmlDeviceGetHandleByIndex(0, &device));
    nvmlMemory_t memory;
    checkError(nvmlDeviceGetMemoryInfo(device, &memory));
    printf("Free memory = %zu; Total memory = %zu\n", memory.free, memory.total);
    return memory.free;
}
