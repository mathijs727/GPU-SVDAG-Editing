#pragma once
#include "cuda_error_check.h"
#include "typedefs.h"
#include <atomic>
#include <gsl-lite/gsl-lite.hpp>
#include <list>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef ENABLE_VULKAN_MEMORY_ALLOCATOR
#include <vk_mem_alloc.h>
#endif

enum class EMemoryType {
    GPU_Managed,
    GPU_Malloc,
    GPU_Async,
    CPU,
    Unknown
};

namespace D3D12MA {
class VirtualBlock;
}

class Memory {
public:
    template <typename T>
    static T* malloc(const char* name, size_t size, EMemoryType type)
    {
        PROFILE_FUNCTION();
        checkAlways(size % sizeof(T) == 0);
        std::lock_guard<std::mutex> guard(singleton.mutex);
        return reinterpret_cast<T*>(singleton.malloc_impl(size, 0, name, type));
    }
    template <typename T>
    static T* malloc(const char* name, size_t size, size_t alignment, EMemoryType type)
    {
        PROFILE_FUNCTION();
        checkAlways(size % sizeof(T) == 0);
        std::lock_guard<std::mutex> guard(singleton.mutex);
        return reinterpret_cast<T*>(singleton.malloc_impl(size, alignment, name, type));
    }
    template <typename T>
    static void free(T* ptr)
    {
        PROFILE_FUNCTION();
        std::lock_guard<std::mutex> guard(singleton.mutex);
        singleton.free_impl(reinterpret_cast<void*>(ptr));
    }
    template <typename T>
    static void realloc(T& ptr, size_t newSize, size_t newAlignment = 0)
    {
        PROFILE_FUNCTION();
        std::lock_guard<std::mutex> guard(singleton.mutex);
        void* copy = reinterpret_cast<void*>(ptr);
        singleton.realloc_impl(copy, newSize, newAlignment);
        ptr = reinterpret_cast<T>(copy);
    }
    template <typename T>
    static void cuda_memcpy(T* dst, const T* src, uint64_t Size, cudaMemcpyKind memcpyKind)
    {
        check(Size % sizeof(T) == 0);
        singleton.cuda_memcpy_impl(reinterpret_cast<uint8*>(dst), reinterpret_cast<const uint8*>(src), Size, memcpyKind);
    }

    template <typename T>
    static const char* get_alloc_name(const T* ptr)
    {
        std::lock_guard<std::mutex> guard(singleton.mutex);
        return singleton.get_alloc_name_impl(reinterpret_cast<void*>(const_cast<T*>(ptr)));
    }
    static std::string get_stats_string()
    {
        std::lock_guard<std::mutex> guard(singleton.mutex);
        return singleton.get_stats_string_impl();
    }

public:
    inline static size_t get_cpu_allocated_memory()
    {
        return singleton.totalAllocatedCPUMemory;
    }
    inline static size_t get_gpu_used_memory()
    {
        return singleton.totalUsedGPUMemory;
    }
    inline static size_t get_gpu_allocated_memory()
    {
        return singleton.totalAllocatedGPUMemory;
    }
    inline static size_t get_gpu_peak_allocated_memory()
    {
        return singleton.peakAllocatedGPUMemory;
    }

    inline static size_t get_cxx_cpu_allocated_memory()
    {
        return singleton.cxxCPUMemory.load();
    }

public:
    inline static void track_add_memory(size_t count)
    {
        singleton.cxxCPUMemory += count;
    }
    inline static void track_del_memory(size_t count)
    {
        singleton.cxxCPUMemory -= count;
    }

private:
    Memory() = default;
    ~Memory();

    Memory(Memory const&) = delete;
    Memory& operator=(Memory const&) = delete;

    void cuda_memcpy_impl(uint8* dst, const uint8* src, uint64 size, cudaMemcpyKind memcpyKind);
    void* malloc_impl(size_t size, size_t newAlignment, const char* name, EMemoryType type);

    void free_impl(void* ptr);
    void realloc_impl(void*& ptr, size_t newSize, size_t newAlignment);
    const char* get_alloc_name_impl(void* ptr) const;
    std::string get_stats_string_impl() const;

    size_t availableDeviceMemory() const;

    struct Element {
        const char* name = nullptr;
        EMemoryType type;
        size_t size = size_t(-1);
    };

    std::mutex mutex;
    size_t totalAllocatedCPUMemory = 0;
    size_t totalAllocatedGPUMemory = 0;
    size_t totalUsedGPUMemory = 0;
    size_t peakAllocatedGPUMemory = 0;
    std::unordered_map<void*, Element> allocations;
#ifdef _WIN32
    std::unordered_set<void*> alignedMallocPointers;
#endif

    std::atomic<size_t> cxxCPUMemory { 0 };

#ifdef ENABLE_VULKAN_MEMORY_ALLOCATOR
    struct MemoryBlock {
        VmaVirtualBlock block;
        uint8_t* pMemory;
        uint32_t numAllocations = 0;
        uint32_t sizeInBytes;
    };
    class VmaBlockAllocator {
    public:
        VmaBlockAllocator(EMemoryType memoryType);
        ~VmaBlockAllocator();

        void* malloc(size_t size, size_t alignment, cudaError_t& error);
        bool free(void*);

    private:
        std::list<MemoryBlock> m_blocks;
        std::unordered_map<void*, std::pair<typename decltype(m_blocks)::iterator, VmaVirtualAllocation>> m_allocations;
        size_t m_currentBlockIdx = 0;
        EMemoryType m_memoryType;
    };
    class VmaAllocator {
    public:
        VmaAllocator(EMemoryType memoryType, size_t memoryPoolSize);
        ~VmaAllocator();

        void* malloc(size_t size, size_t alignment, cudaError_t& error);
        bool free(void*);

    private:
        std::byte* m_pData;
        VmaVirtualBlock m_vmaBlock;
        std::unordered_map<void*, VmaVirtualAllocation> m_allocations;
    };
    VmaBlockAllocator managedAllocator { EMemoryType::GPU_Managed }; // Should not be used much...
    // VmaBlockAllocator gpuAllocator { EMemoryType::GPU_Malloc }
    VmaAllocator gpuAllocator { EMemoryType::GPU_Malloc, availableDeviceMemory() - 1024llu * 1024 * 1024 }; // Free device memory minus 1GiB
#endif

    static Memory singleton;
};

class GpuMemoryPool {
public:
#if 1
    static GpuMemoryPool create(cudaStream_t stream, EMemoryType memoryType = EMemoryType::GPU_Async)
    {
        GpuMemoryPool out {};
        out.memoryType = memoryType;
        return out;
    }
    void release() { }

    template <typename T>
    inline T* mallocAsync()
    {
        return Memory::malloc<T>("GpuMemoryPool::mallocAsync", sizeof(T), memoryType);
    }
    template <typename T>
    inline void freeAsync(T* ptr)
    {
        Memory::free(ptr);
    }

    template <typename T>
    inline gsl::span<T> mallocAsync(size_t count)
    {
        T* pOut = Memory::malloc<T>("GpuMemoryPool::mallocAsync", count * sizeof(T), memoryType);
        return gsl::make_span(pOut, count);
    }
    template <typename T>
    inline void freeAsync(gsl::span<T> items)
    {
        Memory::free(items.data());
    }
#else
    static GpuMemoryPool create(cudaStream_t stream)
    {
        GpuMemoryPool out;
        out.stream = stream;
        cudaMemPoolProps poolProps {};
        poolProps.allocType = cudaMemAllocationType::cudaMemAllocationTypePinned;
        poolProps.handleTypes = cudaMemAllocationHandleType::cudaMemHandleTypeNone;
        poolProps.location.type = cudaMemLocationType::cudaMemLocationTypeDevice;
        poolProps.location.id = 0;
        cudaMemPoolCreate(&out.memPool, &poolProps);
        CUDA_CHECK_ERROR();
        return out;
    }
    void release()
    {
        // ...
        cudaMemPoolDestroy(memPool);
    }

    template <typename T>
    inline T* mallocAsync()
    {
        T* out;
        cudaMallocFromPoolAsync(&out, sizeof(T), memPool, stream);
        CUDA_CHECK_ERROR();
        return out;
    }
    template <typename T>
    inline void freeAsync(T* ptr)
    {
        PROFILE_FUNCTION();
        cudaFreeAsync(ptr, stream);
        CUDA_CHECK_ERROR();
    }

    template <typename T>
    inline gsl::span<T> mallocAsync(size_t count)
    {
        T* out;
        cudaMallocFromPoolAsync(&out, count * sizeof(T), memPool, stream);
        CUDA_CHECK_ERROR();
        return gsl::span<T>(out, count);
    }
    template <typename T>
    inline void freeAsync(gsl::span<T> items)
    {
        cudaFreeAsync(items.data(), stream);
        CUDA_CHECK_ERROR();
    }
#endif

private:
    EMemoryType memoryType;
    cudaStream_t stream;
    cudaMemPool_t memPool;
};

template <typename T, EMemoryType OldMemoryType, EMemoryType NewMemoryType, bool NewAsyncMalloc = false>
gsl::span<T> copySpan(gsl::span<const T> old, const char* name)
{
    gsl::span<uint32_t> newBlock = NewAsyncMalloc ? cudaMallocAsyncRange<T>(old.size(), nullptr) : mallocRange<T>(name, old.size(), NewMemoryType);
    if constexpr (OldMemoryType == EMemoryType::CPU && NewMemoryType == EMemoryType::CPU) {
        memcpy(newBlock.data(), old.data(), old.size_bytes());
    } else {
        cudaMemcpy(newBlock.data(), old.data(), old.size_bytes(), cudaMemcpyDefault);
    }
    return newBlock;
}
