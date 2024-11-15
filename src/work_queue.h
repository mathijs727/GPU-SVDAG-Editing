#pragma once
#include "array.h"
#include "cuda_helpers_cpp.h"
#include "memory.h"
#include "typedefs.h"
#include <cstdint>
#include <cuda.h>
#include <optional>

template <typename WorkItem>
class PingPongQueue {
public:
    static PingPongQueue allocate(uint32_t maxSize, const WorkItem& initialItem)
    {
        constexpr auto memoryType = EMemoryType::GPU_Managed;
        PingPongQueue out {};
        out.m_inputItems = gsl::span(Memory::malloc<WorkItem>("OutputQueue::m_inputItems", maxSize * sizeof(WorkItem), memoryType), maxSize);
        out.m_outputItems = gsl::span(Memory::malloc<WorkItem>("OutputQueue::m_outputItems", maxSize * sizeof(WorkItem), memoryType), maxSize);
        out.m_currentInput = Memory::malloc<uint32_t>("OutputQueue::m_currentInput", sizeof(uint32_t), memoryType);
        out.m_currentOutput = Memory::malloc<uint32_t>("OutputQueue::m_currentOutput", sizeof(uint32_t), memoryType);
        out.m_inputSize = Memory::malloc<uint32_t>("OutputQueue::m_inputSize", sizeof(uint32_t), memoryType);
        cudaMemset(out.m_currentInput, 0, sizeof(uint32_t));
        cudaMemset(out.m_currentOutput, 0, sizeof(uint32_t));
        // cudaMemset(out.m_inputSize, 0, sizeof(uint32_t));
        uint32_t inputSize = 1;
        cudaMemcpy(out.m_inputSize, &inputSize, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(out.m_inputItems.data(), &initialItem, sizeof(WorkItem), cudaMemcpyHostToDevice);
        return out;
    }
    void free()
    {
        Memory::free(m_inputItems.data());
        Memory::free(m_outputItems.data());
        Memory::free(m_currentInput);
        Memory::free(m_currentOutput);
        Memory::free(m_inputSize);
    }
    void swap()
    {
        std::swap(m_inputItems, m_outputItems);
        cudaMemcpy(m_inputSize, m_currentOutput, sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        cudaMemset(m_currentInput, 0, sizeof(uint32_t));
        cudaMemset(m_currentOutput, 0, sizeof(uint32_t));
    }

    static PingPongQueue allocateAsync(uint32_t maxSize, const WorkItem& initialItem, cudaStream_t stream = 0)
    {
        PingPongQueue out {};
        out.m_inputItems = cudaMallocAsyncRange<WorkItem>(maxSize, stream);
        out.m_outputItems = cudaMallocAsyncRange<WorkItem>(maxSize, stream);
        cudaMallocAsync(&out.m_currentInput, sizeof(uint32_t), stream);
        cudaMallocAsync(&out.m_currentOutput, sizeof(uint32_t), stream);
        cudaMallocAsync(&out.m_inputSize, sizeof(uint32_t), stream);
        cudaMemsetAsync(out.m_currentInput, 0, sizeof(uint32_t), stream);
        cudaMemsetAsync(out.m_currentOutput, 0, sizeof(uint32_t), stream);
        uint32_t inputSize = 1;
        cudaMemcpyAsync(out.m_inputSize, &inputSize, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(out.m_inputItems.data(), &initialItem, sizeof(WorkItem), cudaMemcpyHostToDevice, stream);
        return out;
    }
    void freeAsync(cudaStream_t stream = 0)
    {
        cudaFreeAsync(m_inputItems.data(), stream);
        cudaFreeAsync(m_outputItems.data(), stream);
        cudaFreeAsync(m_currentInput, stream);
        cudaFreeAsync(m_currentOutput, stream);
        cudaFreeAsync(m_inputSize, stream);
    }
    void swapAsync(cudaStream_t stream = 0)
    {
        std::swap(m_inputItems, m_outputItems);
        cudaMemcpyAsync(m_inputSize, m_currentOutput, sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(m_currentInput, 0, sizeof(uint32_t), stream);
        cudaMemsetAsync(m_currentOutput, 0, sizeof(uint32_t), stream);
    }
    void swapAndGrowAsync(uint32_t growFactor, cudaStream_t stream = 0)
    {
        std::swap(m_inputItems, m_outputItems);
        cudaMemcpyAsync(m_inputSize, m_currentOutput, sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(m_currentInput, 0, sizeof(uint32_t), stream);
        cudaMemsetAsync(m_currentOutput, 0, sizeof(uint32_t), stream);

        // Each input item may generate up to growFactor output items.
        // Read the current size to the CPU (blocking) and reallocate if the output is too small.
        uint32_t inputSize = 0;
        cudaMemcpy(&inputSize, m_inputSize, sizeof(inputSize), cudaMemcpyDeviceToHost);
        const auto requiredOutputSize = inputSize * growFactor;
        if (requiredOutputSize > m_outputItems.size()) {
            cudaFreeAsync(m_outputItems.data(), stream);
            m_outputItems = cudaMallocAsyncRange<WorkItem>(requiredOutputSize, stream);
        }
    }

    std::vector<WorkItem> getInputItems()
    {
        uint32_t inputSize;
        cudaMemcpy(&inputSize, m_inputSize, sizeof(uint32_t), cudaMemcpyDeviceToHost);

        std::vector<WorkItem> out((size_t)inputSize);
        cudaMemcpy(out.data(), m_inputItems.data(), inputSize * sizeof(WorkItem), cudaMemcpyDeviceToHost);
        return out;
    }
    void setInputItems(gsl::span<const WorkItem> items)
    {
        uint32_t inputSize = (uint32_t)items.size();
        cudaMemcpy(m_inputSize, &inputSize, sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(m_inputItems.data(), items.data(), items.size_bytes(), cudaMemcpyHostToDevice);
    }

    HOST_DEVICE bool consume(WorkItem& outWorkItem)
    {
#ifdef __CUDA_ARCH__
        const uint32_t index = atomicAdd(m_currentInput, 1);
        if (index < (*m_inputSize)) {
            outWorkItem = m_inputItems[index];
            return true;
        } else {
            return false;
        }
#else
        const uint32_t index = (*m_currentInput)++;
        if (index < (*m_inputSize)) {
            outWorkItem = m_inputItems[index];
            return true;
        } else {
            return false;
        }
#endif
    }

    HOST_DEVICE void produce(const WorkItem& item)
    {
#ifdef __CUDA_ARCH__
        const uint32_t index = atomicAdd(m_currentOutput, 1);
        if (index >= m_outputItems.size()) {
            printf("workQueue out-of-bounds: %u > %lu\n", index, m_outputItems.size());
        } else {
            m_outputItems[index] = item;
        }
#else
        const uint32_t index = (*m_currentOutput)++;
        m_outputItems[index] = item;
#endif
    }

private:
    gsl::span<WorkItem> m_inputItems, m_outputItems;
    uint32_t *m_currentInput, *m_currentOutput, *m_inputSize;
};

template <typename T>
struct PushBuffer {
public:
    StaticArray<T> buffer;
    uint32_t* cur;

public:
    HOST static PushBuffer allocate(size_t sizeInItems, EMemoryType memoryType = EMemoryType::GPU_Malloc)
    {
        PushBuffer out {};
        out.buffer = StaticArray<T>::allocate("PushBuffer::buffer", sizeInItems, memoryType);
        out.cur = Memory::malloc<uint32_t>("PushBuffer::cur", sizeof(uint32_t), memoryType);
        cudaMemset(out.cur, 0, sizeof(uint32_t));
        return out;
    }
    HOST void free()
    {
        buffer.free();
        Memory::free(cur);
    }

    HOST_DEVICE void push(const T& item)
    {
#ifdef __CUDA_ARCH__
        const uint32_t offset = atomicAdd(cur, 1);
        check(offset < buffer.size());
        buffer[offset] = item;
#else
        buffer[*cur] = item;
        (*cur)++;
#endif
    }
    HOST_DEVICE gsl::span<T> pushN(size_t count)
    {
#ifdef __CUDA_ARCH__
        const uint32_t offset = atomicAdd(cur, (uint32_t)count);
        check(offset + count <= buffer.size());
        return buffer.span().subspan(offset, count);
#else
        const auto out = buffer.span().subspan(*cur, count);
        *cur += (uint32_t)count;
        return out;
#endif
    }

    HOST_DEVICE gsl::span<T> span()
    {
#ifdef __CUDA_ARCH__
        return static_cast<gsl::span<T>>(buffer).subspan(0, *cur);
#else
        uint32_t numItems;
        cudaMemcpy(&numItems, cur, sizeof(numItems), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
        return static_cast<gsl::span<T>>(buffer).subspan(0, numItems);
#endif
    }

    HOST_DEVICE gsl::span<const T> span() const
    {
#ifdef __CUDA_ARCH__
        return static_cast<gsl::span<const T>>(buffer).subspan(0, *cur);
#else
        uint32_t numItems;
        cudaMemcpy(&numItems, cur, sizeof(numItems), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
        return static_cast<gsl::span<const T>>(buffer).subspan(0, numItems);
#endif
    }

    HOST auto begin()
    {
        return std::begin(buffer);
    }
    HOST auto end()
    {
        return std::begin(buffer) + *cur;
    }
};
