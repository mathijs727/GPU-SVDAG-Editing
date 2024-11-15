#pragma once
#include <cstdint>

// NOTE(Mathijs): DO NOT rename this file to math.h, it will cause problems on Windows.

inline constexpr uint32_t pow2(uint32_t power)
{
    uint32_t out = 1;
    for (uint32_t i = 0; i < power; ++i)
        out *= 2;
    return out;
}
