#pragma once
#include <cstdint>

template <uint32_t ItemSizeInU32_>
struct MyGpuDagItem {
    static constexpr uint32_t ItemSizeInU32 = ItemSizeInU32_;
    uint32_t padding[ItemSizeInU32_];

    // CUDA C++20 I'm waiting, sigh...
    // constexpr auto operator<=>(const DAGElement<ItemSizeInU32>&) const noexcept = default;
    constexpr bool operator==(const MyGpuDagItem<ItemSizeInU32>& rhs) const noexcept
    {
        for (uint32_t i = 0; i < ItemSizeInU32; ++i) {
            if (padding[i] != rhs.padding[i])
                return false;
        }
        return true;
    }
    constexpr bool operator!=(const MyGpuDagItem<ItemSizeInU32>& rhs) const noexcept
    {
        return !(*this == rhs);
    }
    constexpr bool operator<(const MyGpuDagItem<ItemSizeInU32>& rhs) const noexcept
    {
        for (uint32_t i = 0; i < ItemSizeInU32; ++i) {
            if (padding[i] < rhs.padding[i])
                return true;
            else if (padding[i] > rhs.padding[i])
                return false;
            // else: check next uint32_t...
        }
        return false;
    }
};
static_assert(sizeof(MyGpuDagItem<2>) == 2 * sizeof(uint32_t));
static_assert(sizeof(MyGpuDagItem<3>) == 3 * sizeof(uint32_t));
static_assert(sizeof(MyGpuDagItem<4>) == 4 * sizeof(uint32_t));
