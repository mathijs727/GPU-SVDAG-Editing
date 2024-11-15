#pragma once
#include "array.h"
#include "binary_reader.h"
#include "binary_writer.h"
#include "typedefs.h"

template <typename T>
struct StaticArray2D {
    static StaticArray2D allocate(const char* name, uint32_t width, uint32_t height, EMemoryType memoryType = EMemoryType::GPU_Malloc)
    {
        StaticArray2D out {};
        out.width = width;
        out.height = height;
        out.buffer = StaticArray<T>::allocate(name, width * height, memoryType);
        return out;
    }
    inline void free()
    {
        buffer.free();
    }

    HOST StaticArray2D copy(EMemoryType newMemoryType) const
    {
        PROFILE_FUNCTION();
        auto result = allocate(Memory::get_alloc_name(data()), width, height, newMemoryType);
        Memory::cuda_memcpy(result.data(), data(), width * height * sizeof(T), cudaMemcpyDefault);
        return result;
    }

    HOST_DEVICE void write(uint32_t x, uint32_t y, const T& value)
    {
        buffer[y * width + x] = value;
    }
    HOST_DEVICE void write(uint2 pixel, const T& value)
    {
        write(pixel.x, pixel.y, value);
    }

    HOST_DEVICE T read(uint32_t x, uint32_t y) const
    {
        return buffer[y * width + x];
    }
    HOST_DEVICE T read(uint2 pixel) const
    {
        return read(pixel.x, pixel.y);
    }

    HOST_DEVICE T* getPixelPointer(uint32_t x, uint32_t y)
    {
        return &buffer[y * width + x];
    }
    HOST_DEVICE T* getPixelPointer(uint2 pixel)
    {
        return getPixelPointer(pixel.x, pixel.y);
    }
    HOST_DEVICE const T* getPixelPointer(uint32_t x, uint32_t y) const
    {
        return &buffer[y * width + x];
    }
    HOST_DEVICE const T* getPixelPointer(uint2 pixel) const
    {
        return getPixelPointer(pixel.x, pixel.y);
    }

    HOST_DEVICE T& getPixel(uint32_t x, uint32_t y)
    {
        return buffer[y * width + x];
    }
    HOST_DEVICE T& getPixel(uint2 pixel)
    {
        return getPixel(pixel.x, pixel.y);
    }
    HOST_DEVICE const T& getPixel(uint32_t x, uint32_t y) const
    {
        return buffer[y * width + x];
    }
    HOST_DEVICE const T& getPixel(uint2 pixel) const
    {
        return getPixel(pixel.x, pixel.y);
    }

    HOST_DEVICE auto begin()
    {
        return buffer.begin();
    }
    HOST_DEVICE auto end()
    {
        return buffer.end();
    }
    HOST_DEVICE T* data()
    {
        return buffer.data();
    }
    HOST_DEVICE const T* data() const
    {
        return buffer.data();
    }

    HOST StaticArray2D<T> create_gpu() const
    {
        StaticArray2D out;
        out.buffer = buffer.create_gpu();
        out.width = width;
        out.height = height;
        return out;
    }

    HOST_DEVICE size_t size_in_bytes() const
    {
        return width * height * sizeof(T);
    }

    void writeTo(BinaryWriter& writer) const
    {
        writer.write(buffer);
        writer.write(width);
        writer.write(height);
    }
    void readFrom(BinaryReader& reader)
    {
        reader.read(buffer);
        reader.read(width);
        reader.read(height);
    }

    StaticArray<T> buffer;
    uint32_t width, height;
};
