#pragma once
#include "binary_reader.h"
#include "binary_writer.h"
#include "cuda_error_check.h"
#include "memory.h"
#include "my_units.h"
#include "typedefs.h"
#include "utils.h"
#include <gsl/gsl-lite.hpp>
#include <type_traits>

template <typename T, typename Size = uint64, size_t Alignment = 0>
struct StaticArray {
public:
    using value_type = T;

public:
    StaticArray() = default;
    HOST_DEVICE StaticArray(T* arrayData, Size arraySize, EMemoryType memoryType)
        : arrayData(arrayData)
        , arraySize(arraySize)
        , memoryType(memoryType)
    {
    }
    HOST_DEVICE StaticArray(decltype(nullptr))
        : StaticArray(nullptr, 0, EMemoryType::CPU)
    {
    }

    void writeTo(BinaryWriter& writer) const
    {
        writer.write(arraySize);
        writer.write(memoryType);

        if (memoryType == EMemoryType::CPU) {
            if constexpr (has_write_to<T>) {
                for (size_t i = 0; i < arraySize; ++i) {
                    writer.write(arrayData[i]);
                }
            } else {
                writer.writeRaw(arrayData, arraySize);
            }
        } else {
            std::vector<T> arrayDataCPU((size_t)arraySize);
            cudaMemcpy(arrayDataCPU.data(), arrayData, arraySize * sizeof(T), cudaMemcpyDeviceToHost);
            if constexpr (has_write_to<T>) {
                for (size_t i = 0; i < arraySize; ++i) {
                    writer.write(arrayDataCPU[i]);
                }
            } else {
                writer.writeRaw(arrayDataCPU.data(), arraySize);
            }
        }
    }
    void readFrom(BinaryReader& reader)
    {
        reader.read(arraySize);
        reader.read(memoryType);
        if (arraySize > 0) {
            arrayData = Memory::malloc<T>("StaticArray::readFrom", arraySize * sizeof(T), Alignment, memoryType);
        }

        if (memoryType == EMemoryType::CPU) {
            if constexpr (has_read_from<T>) {
                for (size_t i = 0; i < arraySize; ++i) {
                    reader.read(arrayData[i]);
                }
            } else {
                reader.readRaw(arrayData, arraySize);
            }
        } else {
            std::vector<T> arrayDataCPU((size_t)arraySize);
            if constexpr (has_read_from<T>) {
                for (size_t i = 0; i < arraySize; ++i) {
                    reader.read(arrayDataCPU[i]);
                }
            } else {
                reader.readRaw(arrayDataCPU.data(), arraySize);
            }
            cudaMemcpy(arrayData, arrayDataCPU.data(), arraySize * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    static StaticArray<T, Size, Alignment> allocate(const char* name, Size arraySize, EMemoryType type)
    {
        PROFILE_FUNCTION();
        T* ptr = Memory::malloc<T>(name, (arraySize + padding_in_items) * sizeof(T), Alignment, type);
        return { ptr, arraySize, type };
    }
    static StaticArray<T, Size, Alignment> allocate(const char* name, gsl::span<const T> array, EMemoryType type)
    {
        PROFILE_FUNCTION();
        auto out = allocate(name, array.size(), type);
        if (type == EMemoryType::GPU_Async)
            cudaMemcpyAsync(out.data(), array.data(), array.size_bytes(), cudaMemcpyDefault, nullptr);
        else
            cudaMemcpy(out.data(), array.data(), array.size_bytes(), cudaMemcpyDefault);
        return out;
    }

    HOST size_t size_in_bytes() const
    {
        return arraySize * sizeof(T);
    }

    HOST StaticArray<T, Size> create_gpu() const
    {
        PROFILE_FUNCTION();
        check(is_valid());
        auto result = allocate(Memory::get_alloc_name(data()), size(), EMemoryType::GPU_Malloc);
        copy_to_gpu_strict(result);
        return result;
    }

    HOST void copy_to_gpu_strict(StaticArray<T, Size>& gpuArray) const
    {
        PROFILE_FUNCTION();
        check(gpuArray.size() == size());
        Memory::cuda_memcpy(gpuArray.data(), data(), size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    HOST void upload_to_gpu(gsl::span<const T> cpuArray)
    {
        PROFILE_FUNCTION();
        check(cpuArray.size() == size());
        Memory::cuda_memcpy(data(), cpuArray.data(), cpuArray.size_bytes(), cudaMemcpyHostToDevice);
    }
    HOST void upload_to_gpu()
    {
        PROFILE_FUNCTION();
        auto newArray = copy(EMemoryType::GPU_Malloc);
        free();
        *this = newArray;
    }

    HOST std::vector<T> copy_to_cpu() const
    {
        PROFILE_FUNCTION();
        std::vector<T> out((size_t)size());
        if (memoryType == EMemoryType::CPU) {
            memcpy(out.data(), data(), size() * sizeof(T));
        } else if (memoryType == EMemoryType::GPU_Async) {
            cudaMemcpyAsync(out.data(), data(), size() * sizeof(T), cudaMemcpyDeviceToHost, nullptr);
        } else {
            Memory::cuda_memcpy(out.data(), data(), size() * sizeof(T), cudaMemcpyDeviceToHost);
        }
        return out;
    }

    HOST StaticArray<T, Size, Alignment> copy() const
    {
        PROFILE_FUNCTION();
        check(is_valid());
        auto result = StaticArray<T, Size, Alignment>::allocate(Memory::get_alloc_name(data()), size(), memoryType);
        Memory::cuda_memcpy(result.data(), data(), size() * sizeof(T), cudaMemcpyDefault);
        return result;
    }
    HOST StaticArray<T, Size, Alignment> copy(EMemoryType newMemoryType) const
    {
        PROFILE_FUNCTION();
        check(is_valid());
        auto result = StaticArray<T, Size, Alignment>::allocate(Memory::get_alloc_name(data()), size(), newMemoryType);
        Memory::cuda_memcpy(result.data(), data(), size() * sizeof(T), cudaMemcpyDefault);
        return result;
    }

    HOST void free()
    {
        PROFILE_FUNCTION();
        Memory::free(arrayData);
        reset();
    }
    HOST void reset()
    {
        *this = nullptr;
    }

    HOST_DEVICE Size size() const
    {
        return arraySize;
    }
    HOST_DEVICE bool empty() const
    {
        return arraySize > 0;
    }
    HOST_DEVICE const T* data() const
    {
        return arrayData;
    }
    HOST_DEVICE T* data()
    {
        return arrayData;
    }

    HOST_DEVICE bool is_valid() const
    {
        return arrayData != nullptr;
    }

    HOST_DEVICE bool is_valid_index(Size index) const
    {
        return index < arraySize;
    }

    HOST_DEVICE my_units::bytes memory_allocated() const
    {
        return my_units::bytes(arraySize * sizeof(T));
    }
    HOST_DEVICE my_units::bytes memory_used() const
    {
        return my_units::bytes(arraySize * sizeof(T));
    }

    HOST_DEVICE T& operator[](Size index)
    {
        if constexpr (std::is_same_v<Size, uint64_t>) {
            checkf(index < arraySize, "invalid index: %" PRIu64 " for size %" PRIu64, index, arraySize);
        } else {
            checkf(index < arraySize, "invalid index: %u for size %u", index, arraySize);
        }
        return arrayData[index];
    }
    HOST_DEVICE const T& operator[](Size index) const
    {
        if constexpr (std::is_same_v<Size, uint64_t>) {
            checkf(index < arraySize, "invalid index: %" PRIu64 " for size %" PRIu64, index, arraySize);
        } else {
            checkf(index < arraySize, "invalid index: %u for size %u", index, arraySize);
        }
        return arrayData[index];
    }
    HOST_DEVICE bool operator==(StaticArray<T> other) const
    {
        return other.arrayData == arrayData && other.arraySize == arraySize;
    }
    HOST_DEVICE bool operator!=(StaticArray<T> other) const
    {
        return other.arrayData != arrayData || other.arraySize != arraySize;
    }

    HOST_DEVICE const T* begin() const
    {
        return arrayData;
    }
    HOST_DEVICE const T* end() const
    {
        return arrayData + arraySize;
    }

    HOST_DEVICE T* begin()
    {
        return arrayData;
    }
    HOST_DEVICE T* end()
    {
        return arrayData + arraySize;
    }

    HOST_DEVICE operator gsl::span<T>()
    {
        return gsl::span(this->arrayData, this->arraySize);
    }
    HOST_DEVICE operator gsl::span<const T>() const
    {
        return gsl::span(this->arrayData, this->arraySize);
    }
    HOST_DEVICE gsl::span<T> span()
    {
        return gsl::span(this->arrayData, this->arraySize);
    }
    HOST_DEVICE gsl::span<const T> span() const
    {
        return gsl::span(this->arrayData, this->arraySize);
    }
    HOST_DEVICE gsl::span<const T> cspan() const
    {
        return gsl::span(this->arrayData, this->arraySize);
    }

protected:
    T* arrayData = nullptr;
    Size arraySize = 0;

    static constexpr Size padding_in_items = 32;

public:
    EMemoryType memoryType = EMemoryType::Unknown;
};

template <typename T, typename Size = uint64, size_t Alignment = 0>
struct DynamicArray : StaticArray<T, Size, Alignment> {
public:
    DynamicArray() = default;
    HOST_DEVICE DynamicArray(T* arrayData, Size arraySize, EMemoryType memoryType)
        : StaticArray<T, Size, Alignment>(arrayData, arraySize, memoryType)
        , allocatedSize(arraySize)
    {
    }
    HOST_DEVICE DynamicArray(EMemoryType memoryType)
        : StaticArray<T, Size, Alignment>(nullptr, 0, memoryType)
        , allocatedSize(0)
    {
    }
    HOST_DEVICE DynamicArray(StaticArray<T, Size, Alignment> array)
        : StaticArray<T, Size, Alignment>(array)
        , allocatedSize(array.size())
    {
    }

public:
    void writeTo(BinaryWriter& writer) const
    {
        StaticArray<T, Size>::writeTo(writer);
        writer.write(allocatedSize);
    }

    void readFrom(BinaryReader& reader)
    {
        StaticArray<T, Size>::readFrom(reader);
        allocatedSize = this->arraySize;

        Size desiredAllocatedSize;
        reader.read(desiredAllocatedSize);
        if (this->arraySize != desiredAllocatedSize)
            reserve(desiredAllocatedSize);
    }

    HOST void copy_to_gpu_flexible(DynamicArray<T, Size>& gpuArray) const
    {
        PROFILE_FUNCTION_SLOW();

        if (!gpuArray.is_valid())
            gpuArray = this->allocate(Memory::get_alloc_name(this->data()), 2 * this->allocated_size(), EMemoryType::GPU_Malloc);

        if (gpuArray.allocated_size() < this->size())
            gpuArray.reserve((size_t)(1.2 * (double)this->size()));
        gpuArray.arraySize = this->size();
        Memory::cuda_memcpy(gpuArray.data(), this->data(), this->size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    HOST_DEVICE Size allocated_size() const
    {
        return allocatedSize;
    }

    HOST_DEVICE my_units::bytes memory_allocated() const
    {
        return my_units::bytes(allocatedSize * sizeof(T));
    }

    HOST void hack_set_size(Size newSize)
    {
        this->arraySize = newSize;
    }

    HOST void push_back(T element)
    {
        check(this->arraySize <= allocatedSize);
        if (this->arraySize == allocatedSize) {
            reserve(2 * allocatedSize + 1); // Double the storage
        }

        check(this->arraySize <= allocatedSize);
        if (this->memoryType == EMemoryType::CPU)
            this->arrayData[this->arraySize++] = element;
        else
            cudaMemcpy(&this->arrayData[this->arraySize++], &element, sizeof(T), cudaMemcpyHostToDevice);
    }
    HOST const T& back() const
    {
        return this->arrayData[this->arraySize - 1];
    }
    HOST T& back()
    {
        return this->arrayData[this->arraySize - 1];
    }
    HOST Size add(T element)
    {
        const Size oldSize = this->arraySize;
        push_back(element);
        return oldSize;
    }
    template <typename... TArgs>
    HOST Size emplace(TArgs... args)
    {
        return add(T { std::forward<TArgs>(args)... });
    }

    HOST void reserve(Size amount)
    {
        PROFILE_FUNCTION();

        if (amount <= allocatedSize)
            return; // Array is already large enough.

        const Size sizeInBytes = (amount + this->padding_in_items) * sizeof(T);
        if (this->arrayData)
            Memory::realloc(this->arrayData, sizeInBytes, Alignment);
        else
            this->arrayData = Memory::malloc<T>("DynamicArray::reserve", sizeInBytes, Alignment, this->memoryType);
        allocatedSize = amount;
    }

    HOST void resize(Size amount)
    {
        reserve(amount);
        this->arraySize = amount;
        check(this->arraySize <= allocatedSize);
    }

    HOST void shrink()
    {
        PROFILE_FUNCTION();
        check(this->arrayData);
        check(this->arraySize != 0);
        allocatedSize = this->arraySize;
        Memory::realloc(this->arrayData, (allocatedSize + this->padding_in_items) * sizeof(T), Alignment);
    }

protected:
    Size allocatedSize = 0;
};