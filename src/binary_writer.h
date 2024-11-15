#pragma once
#include "is_of_type.h"
#include "memory.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <optional>
#include <type_traits>
#if __cpp_concepts >= 201907L
#include <concepts>
#endif

// Write objects to a binary file.
// Can automatically copy trivially copyable types, std::string, std::filesystem, std::vector and std::variant.
//
// Custom types can be serialized by adding a writeTo(BinaryWriter&) function.
// For example:
// struct MyCustomStruct {
//     std::string variable1;
//     int variable2;
//
//     void writeTo(BinaryWriter& writer) {
//         writer.write(variable1);
//         writer.write(variable2);
//     }
// };

class BinaryWriter {
public:
    BinaryWriter(const std::filesystem::path& filePath);

    template <typename T>
    void writeRaw(const T*, size_t);

    template <typename T>
    void write(const T& src);
    template <typename T>
    void write(const T& src, EMemoryType memoryType);
    template <typename T>
    void write(const std::optional<T>&);

private:
    template <typename T, size_t idx = 0>
    void writeVariant(const T& variant);

private:
    std::ofstream m_fileStream;
};

inline BinaryWriter::BinaryWriter(const std::filesystem::path& filePath)
    : m_fileStream(filePath, std::ios::binary)
{
    assert(std::filesystem::exists(filePath));
    assert(m_fileStream);
}

// clang-format off
template <typename T>
concept has_write_to = requires(const T& item, BinaryWriter& writer)
{
    { item.writeTo(writer) } -> std::same_as<void>;
};
// clang-format on

template <typename T>
inline void BinaryWriter::write(const T& src, EMemoryType memoryType)
{
    if constexpr (std::is_pointer_v<T>) {
        if (memoryType == EMemoryType::CPU) {
            write(*src);
        } else {
            using ItemT = std::remove_pointer_t<T>;

            ItemT srcCPU;
            cudaMemcpy(&srcCPU, src, sizeof(ItemT), cudaMemcpyDeviceToHost);
            write(srcCPU);
        }
    } else {
        write(src);
    }
}

template <typename T>
inline void BinaryWriter::write(const T& src)
{
    static_assert(!std::is_pointer_v<T> && !is_gsl_span<T>::value);

    if constexpr (has_write_to<T>) {
        src.writeTo(*this);
    } else if constexpr (is_std_vector<T>::value) {
        using ItemT = typename T::value_type;
        write(src.size());
        if constexpr (std::is_trivially_copyable_v<ItemT>) {
            m_fileStream.write(reinterpret_cast<const char*>(src.data()), src.size() * sizeof(ItemT));
        } else {
            for (const auto& item : src)
                write(item);
        }
    } else if constexpr (is_std_array<T>::value) {
        for (const auto& item : src)
            write(item);
    } else if constexpr (is_std_variant<T>::value) {
        writeVariant(src);
    } else if constexpr (is_std_optional_v<T>) {
        write(src.has_value());
        if (src.has_value())
            write(src.value());
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        write(src.string());
    } else if constexpr (std::is_same_v<T, std::string>) {
        write(src.size());
        m_fileStream.write(src.data(), src.size());
    } else if constexpr (std::is_trivially_copyable_v<T>) {
        m_fileStream.write(reinterpret_cast<const char*>(&src), sizeof(T));
    } else {
        static_assert(always_false<T>, "Type does not support serialization.");
    }
}

template <typename T>
void BinaryWriter::write(const std::optional<T>& optSrc)
{
    const bool hasValue = optSrc.has_value();
    write(hasValue);
    if (hasValue)
        write(optSrc.value());
}

template <typename T>
void BinaryWriter::writeRaw(const T* items, size_t count)
{
    static_assert(std::is_trivially_copyable_v<T>);
    m_fileStream.write(reinterpret_cast<const char*>(items), (std::streamsize)(count * sizeof(T)));
}

template <typename T, size_t idx>
void BinaryWriter::writeVariant(const T& variant)
{
    if (variant.index() == idx)
        write(std::get<idx>(variant));

    if constexpr (idx + 1 < std::variant_size_v<T>)
        writeVariant<T, idx + 1>(variant);
}
