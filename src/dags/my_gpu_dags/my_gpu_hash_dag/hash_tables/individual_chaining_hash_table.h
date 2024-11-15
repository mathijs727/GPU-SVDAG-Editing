#pragma once
#include "array.h"
#include "cuda_helpers_cpp.h"
#include "dags/hash_dag/hash_dag_globals.h"
#include "safe_cooperative_groups.h"
#include "typedefs.h"
#include "utils.h"
#include <algorithm>
#include <gsl/gsl-lite.hpp>
#include <type_traits>
#include <vector>

template <EMemoryType MemoryType>
class LinearAllocator {
public:
    HOST static LinearAllocator create(uint32_t numItems, uint32_t itemSizeInU32)
    {
        const auto sizeInU32 = numItems * itemSizeInU32;

        LinearAllocator out {};
        out.itemSizeInU32 = itemSizeInU32;
        out.itemsMemory = mallocRange<uint32_t>("LinearAllocator::itemsMemory", sizeInU32, MemoryType);
        out.currentIndex = Memory::malloc<uint32_t>("LinearAllocator index", sizeof(uint32_t), MemoryType);
        if (MemoryType == EMemoryType::CPU)
            *out.currentIndex = 0;
        else
            cudaMemset(out.currentIndex, 0x0, sizeof(uint32_t));
        return out;
    }
    HOST void release()
    {
        Memory::free(itemsMemory.data());
        Memory::free(currentIndex);
    }

    template <EMemoryType NewMemoryType>
    LinearAllocator<NewMemoryType> copy() const
    {
        auto out = LinearAllocator<NewMemoryType>::create((uint32_t)itemsMemory.size() / itemSizeInU32, itemSizeInU32);
        check(out.itemsMemory.size() == itemsMemory.size());
        if constexpr (MemoryType == EMemoryType::CPU && NewMemoryType == EMemoryType::CPU) {
            *out.currentIndex = *currentIndex;
            memcpy(out.itemsMemory.data(), itemsMemory.data(), itemsMemory.size_bytes());
        } else {
            cudaMemcpy(out.currentIndex, currentIndex, sizeof(uint32_t), cudaMemcpyDefault);
            cudaMemcpy(out.itemsMemory.data(), itemsMemory.data(), itemsMemory.size_bytes(), cudaMemcpyDefault);
        }

        return out;
    }

    HOST void writeTo(BinaryWriter& writer) const
    {
        std::vector<uint32_t> itemsMemoryCPU(itemsMemory.size());
        uint32_t currentIndexCPU;
        cudaMemcpy(itemsMemoryCPU.data(), itemsMemory.data(), itemsMemory.size_bytes(), cudaMemcpyDeviceToHost);
        cudaMemcpy(&currentIndexCPU, currentIndex, sizeof(currentIndexCPU), cudaMemcpyDeviceToHost);

        writer.write(itemSizeInU32);
        writer.write(itemsMemoryCPU);
        writer.write(currentIndexCPU);
    }
    HOST void readFrom(BinaryReader& reader)
    {
        uint32_t currentIndexCPU;
        reader.read(this->itemSizeInU32);

        std::vector<uint32_t> itemsMemoryCPU;
        reader.read(itemsMemoryCPU);
        uint32_t* memory = Memory::malloc<uint32_t>("LinearAllocator", itemsMemoryCPU.size() * sizeof(uint32_t), EMemoryType::GPU_Managed);
        cudaMemset(memory, 0, sizeof(uint32_t));
        this->itemsMemory = gsl::span(memory, itemsMemoryCPU.size());
        cudaMemcpy(itemsMemory.data(), itemsMemoryCPU.data(), itemsMemoryCPU.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

        reader.read(currentIndexCPU);
        this->currentIndex = Memory::malloc<uint32_t>("LinearAllocator index", sizeof(uint32_t), EMemoryType::GPU_Managed);
        cudaMemcpy(currentIndex, &currentIndexCPU, sizeof(currentIndexCPU), cudaMemcpyHostToDevice);
    }

    HOST void reserveIfNecessary(size_t additionalSpace)
    {
        constexpr double growFactor = 1.5;

        uint32_t currentIndexCPU;
        if constexpr (MemoryType == EMemoryType::GPU_Malloc)
            cudaMemcpy(&currentIndexCPU, currentIndex, sizeof(currentIndexCPU), cudaMemcpyDeviceToHost);
        else
            currentIndexCPU = *currentIndex;
        const auto requiredSizeInU32 = (currentIndexCPU + additionalSpace) * itemSizeInU32 * 2;
        if (requiredSizeInU32 > itemsMemory.size() - itemSizeInU32) {
            uint32_t newSizeInU32 = (uint32_t)((double)requiredSizeInU32 * growFactor);
            newSizeInU32 = ((newSizeInU32 / itemSizeInU32) + 1) * itemSizeInU32; // Round up to multiple of itemSizeInU32;

            auto newItemsMemory = mallocRange<uint32_t>("LinearAllocator", newSizeInU32, MemoryType);
            if constexpr (MemoryType == EMemoryType::CPU)
                std::copy(std::begin(itemsMemory), std::end(itemsMemory), std::begin(newItemsMemory));
            else
                cudaMemcpy(newItemsMemory.data(), itemsMemory.data(), itemsMemory.size_bytes(), cudaMemcpyDefault);
            Memory::free(itemsMemory.data());
            itemsMemory = newItemsMemory;
        }
    }

    DEVICE void initAsWarp(uint32_t) { }
#ifdef __CUDACC__
    DEVICE uint32_t allocateAsWarp()
    {
        const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        uint32_t out;
        if (warp.thread_rank() == 0) {
            out = atomicAdd(currentIndex, 1);
            check(out * itemSizeInU32 < itemsMemory.size());
        }
        return warp.shfl(out, 0);
    }
#endif

    HOST_DEVICE uint32_t allocate()
    {
#ifdef __CUDA_ARCH__
        const auto out = atomicAdd(currentIndex, 1);
        check(out * itemSizeInU32 < itemsMemory.size());
        return out;
#else
        reserveIfNecessary(1);
        const auto out = *currentIndex;
        *currentIndex += 1;
        check(out * itemSizeInU32 < itemsMemory.size());
        return out;
#endif
    }
    HOST_DEVICE void free(uint32_t) { }

    HOST_DEVICE bool isValidPointer(uint32_t itemIdx) const
    {
        return itemIdx * itemSizeInU32 < itemsMemory.size();
    }
    HOST_DEVICE uint32_t* decodePointer(uint32_t itemIdx)
    {
        check(itemIdx * itemSizeInU32 < itemsMemory.size());
        return &itemsMemory[itemIdx * itemSizeInU32];
    }
    HOST_DEVICE const uint32_t* decodePointer(uint32_t itemIdx) const
    {
        check(itemIdx * itemSizeInU32 < itemsMemory.size());
        return &itemsMemory[itemIdx * itemSizeInU32];
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
        return sizeof(*this) + sizeof(uint32_t) + itemsMemory.size_bytes();
    }
    HOST size_t used_bytes() const
    {
        uint32_t currentIndexCPU;
        if constexpr (MemoryType == EMemoryType::CPU)
            currentIndexCPU = *currentIndex;
        else
            cudaMemcpy(&currentIndexCPU, currentIndex, sizeof(currentIndexCPU), cudaMemcpyDeviceToHost);
        return sizeof(*this) + sizeof(uint32_t) + currentIndexCPU * sizeof(uint32_t);
    }
    HOST uint32_t numItemsUsed() const
    {
        if constexpr (MemoryType != EMemoryType::CPU)
            checkAlways(false);
        return *currentIndex;
    }

    HOST my_units::bytes memory_allocated() const
    {
        return { itemsMemory.size_bytes() };
    }
    HOST my_units::bytes memory_used() const
    {
        return { used_bytes() };
    }

    uint32_t itemSizeInU32;

private:
    template <EMemoryType>
    friend class LinearAllocator;

    gsl::span<uint32_t> itemsMemory;
    uint32_t* currentIndex;
};

template <EMemoryType MemoryType>
class IndividualChainingHashTable {
public:
    uint32_t itemSizeInU32;
    static constexpr uint32_t not_found = (uint32_t)-1;

    using ElementDecoder = const uint32_t*;

public:
    static HOST IndividualChainingHashTable allocate(uint32_t numBuckets, uint32_t numReservedElements, uint32_t itemSizeInU32)
    {
        IndividualChainingHashTable out {};
        out.itemSizeInU32 = itemSizeInU32;
        out.table = StaticArray<uint32_t>::allocate("IndividualChainingHashTable", numBuckets, MemoryType);
        out.allocator = LinearAllocator<MemoryType>::create(numReservedElements, itemSizeInU32 + 1);
        if constexpr (MemoryType == EMemoryType::CPU) {
            std::fill(std::begin(out.table), std::end(out.table), not_found);
        } else {
            // deviceMemset32Async(out.table.data(), not_found, out.table.size(), nullptr);
            // cudaStreamSynchronize(nullptr);
            cudaMemset(out.table.data(), 0xFF, out.table.size_in_bytes());
        }
        return out;
    }
    HOST void free()
    {
        table.free();
        allocator.release();
    }
    template <EMemoryType NewMemoryType>
    HOST IndividualChainingHashTable<NewMemoryType> copy() const
    {
        IndividualChainingHashTable<NewMemoryType> out;
        out.itemSizeInU32 = itemSizeInU32;
        out.table = table.copy(NewMemoryType);
        out.allocator = allocator.template copy<NewMemoryType>();
        return out;
    }

    HOST void reserveIfNecessary(uint32_t numNewItems)
    {
        allocator.reserveIfNecessary(numNewItems);
    }

    uint32_t numBuckets() const
    {
        return (uint32_t)table.size();
    }
    double currentLoadFactor() const
    {
        if constexpr (MemoryType == EMemoryType::CPU) {
            checkAlways(MemoryType == EMemoryType::CPU);
            return (double)allocator.numItemsUsed() / (double)table.size();
        } else {
            return 0.0;
        }
    }

    HOST_DEVICE uint32_t find(const uint32_t* inItem) const
    {
        const auto hash = Utils::murmurhash32xN(inItem, itemSizeInU32);
        const auto hashIndex = hash % table.size();

        uint32_t hashNodeIdx = table[hashIndex];
        while (hashNodeIdx != not_found) {
            const uint32_t* pHashNode = allocator.decodePointer(hashNodeIdx);
            if (Utils::compare_u32_array(inItem, pHashNode, itemSizeInU32))
                return hashNodeIdx;

            hashNodeIdx = pHashNode[itemSizeInU32];
        }
        return not_found;
    }

    HOST_DEVICE uint32_t add(const uint32_t* inItem)
    {
#ifndef __CUDA_ARCH__
        reserveIfNecessary(1);
#endif
        const auto hash = Utils::murmurhash32xN(inItem, itemSizeInU32);
        const auto hashIndex = hash % table.size();
        uint32_t* head = &table[hashIndex];

        const uint32_t elementIdx = allocator.allocate();
        uint32_t* element = allocator.decodePointer(elementIdx);
        memcpy(element, inItem, itemSizeInU32 * sizeof(uint32_t));

        uint32_t* nextElementIdx = element + itemSizeInU32;
        // https://en.cppreference.com/w/cpp/atomic/atomic_compare_exchange
#ifdef __CUDA_ARCH__
        uint32_t oldHead = not_found;
        do {
            *nextElementIdx = oldHead;
        } while ((oldHead = atomicCAS(head, oldHead, elementIdx)) != *nextElementIdx);
#else
        *nextElementIdx = *head;
        *head = elementIdx;
        ++currentSize;
#endif

        return elementIdx;
    }

    HOST_DEVICE uint32_t* decodePointer(uint32_t idx)
    {
        return allocator.decodePointer(idx);
    }
    HOST_DEVICE const uint32_t* decodePointer(uint32_t idx) const
    {
        return allocator.decodePointer(idx);
    }

    HOST void writeTo(BinaryWriter& writer) const
    {
        writer.write(currentSize);
        writer.write(itemSizeInU32);
        writer.write(table);
    }
    HOST void readFrom(BinaryReader& reader)
    {
        reader.read(currentSize);
        reader.read(itemSizeInU32);
        reader.read(table);
    }
    HOST bool is_valid() const
    {
        return table.is_valid();
    }
    my_units::bytes memory_used_by_items() const { return { 0 }; }
    my_units::bytes memory_used_by_table_and_items() const { return { 0 }; }
    my_units::bytes memory_used_by_table_and_slabs() const { return { 0 }; }
    my_units::bytes memory_allocated() const { return table.memory_allocated() + allocator.memory_allocated(); }

private:
    template <EMemoryType>
    friend class IndividualChainingHashTable;

    uint32_t currentSize = 0;
    StaticArray<uint32_t> table;
    LinearAllocator<MemoryType> allocator;
};
