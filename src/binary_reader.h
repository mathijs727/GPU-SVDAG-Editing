#pragma once
#include "cuda_helpers_cpp.h"
#include "is_of_type.h"
#include "memory.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <optional>
#include <type_traits>
#include <vector>
#if __cpp_concepts >= 201907L
#include <concepts>
#endif

class BinaryReader {
public:
    BinaryReader(const std::filesystem::path& filePath);

    template <typename T>
    void readRaw(T*, size_t);

    template <typename T>
    void read(T& dst);
    template <typename T>
    void read(T*& dst, EMemoryType memoryType);
    template <typename T>
    void read(std::optional<T>& dst);

    template <typename T>
    T* read(EMemoryType memoryType);
    template <typename T>
    T read();
    std::vector<std::byte> readBytes(size_t numBytes);

private:
    template <typename T>
    void readVariant(T& dst);
    template <typename T, size_t curIdx = 0>
    void readVariantIdx(T& dst, size_t dstIdx);

private:
    std::ifstream m_fileStream;
};

inline BinaryReader::BinaryReader(const std::filesystem::path& filePath)
    : m_fileStream(filePath, std::ios::binary)
{
    assert(std::filesystem::exists(filePath));
    assert((bool)m_fileStream);
}

template <typename T>
void BinaryReader::readRaw(T* items, size_t count)
{
    static_assert(std::is_trivially_copyable_v<T>);
    m_fileStream.read(reinterpret_cast<char*>(items), (std::streamsize)(count * sizeof(T)));
}

template <typename T>
inline void BinaryReader::read(T& dst)
{
    dst = read<T>();
}
template <typename T>
inline void BinaryReader::read(T*& dst, EMemoryType memoryType)
{
    dst = read<T>(memoryType);
}

template <typename T>
inline void BinaryReader::read(std::optional<T>& dst)
{
    const bool containsValue = read<bool>();
    if (containsValue)
        dst.emplace(read<T>());
}

// clang-format off
template <typename T>
concept has_read_from = requires(T& item, BinaryReader& reader) {
    { item.readFrom(reader) } -> std::same_as<void>;
};
// clang-format on

template <typename T>
inline T* BinaryReader::read(EMemoryType memoryType)
{
    T* out = Memory::malloc<T>("ptr", sizeof(T), memoryType);

    const T item = read<T>();
    if (memoryType == EMemoryType::CPU) {
        *out = item;
    } else {
        cudaMemcpy(out, &item, sizeof(T), cudaMemcpyHostToDevice);
    }
    return out;
}
template <typename T>
inline T BinaryReader::read()
{
    static_assert(!std::is_pointer_v<T> && !is_gsl_span<T>::value);

    if constexpr (has_read_from<T>) {
        T dst {};
        dst.readFrom(*this);
        return dst;
    } else if constexpr (is_std_vector<T>::value) {
        using ItemT = typename T::value_type;
        const size_t vectorLength = read<size_t>();

        T dst;
        if constexpr (std::is_trivially_copyable_v<ItemT>) {
            dst.resize(vectorLength);
            m_fileStream.read(reinterpret_cast<char*>(dst.data()), vectorLength * sizeof(ItemT));
        } else {
            for (size_t i = 0; i < vectorLength; i++)
                dst.push_back(read<ItemT>());
        }
        return dst;
    } else if constexpr (is_std_array<T>::value) {
        using ItemT = typename T::value_type;
        T dst;
        for (auto& item : dst)
            item = read<ItemT>();
        return dst;
    } else if constexpr (is_std_variant<T>::value) {
        T dst;
        readVariant(dst);
        return dst;
    } else if constexpr (is_std_optional_v<T>) {
        if (read<bool>())
            return read<std_optional_type_t<T>>();
        else
            return {};
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        return std::filesystem::path(read<std::string>());
    } else if constexpr (std::is_same_v<T, std::string>) {
        const size_t stringLength = read<size_t>();

        std::string dst;
        dst.resize(stringLength);
        m_fileStream.read(dst.data(), (std::streamsize)dst.size());
        return dst;
    } else if constexpr (std::is_trivially_copyable_v<T>) {
        T dst {};
        m_fileStream.read(reinterpret_cast<char*>(&dst), sizeof(T));
        return dst;
    } else {
        static_assert(always_false<T>, "Type does not support deserialization.");
    }
}

inline std::vector<std::byte> BinaryReader::readBytes(size_t numBytes)
{
    std::vector<std::byte> out;
    out.resize(numBytes);
    m_fileStream.read(reinterpret_cast<char*>(out.data()), (std::streamsize)out.size());
    return out;
}

template <typename T>
void BinaryReader::readVariant(T& dst)
{
    size_t index;
    read(index);

    readVariantIdx<T, 0>(dst, index);
}

template <typename T, size_t curIdx>
void BinaryReader::readVariantIdx(T& dst, size_t dstIdx)
{
    if (curIdx == dstIdx)
        read(std::get<curIdx>(dst));

    if constexpr (curIdx + 1 < std::variant_size_v<T>)
        readVariantIdx<T, curIdx + 1>(dst, dstIdx);
}
