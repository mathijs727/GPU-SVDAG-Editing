#pragma once
#include "array.h"
#include "cuda_math.h"
#include "typedefs.h"
#include <cmath>

namespace ColorUtils {
HOST_DEVICE constexpr uint16 uint3_to_rgb565(uint3 rgb)
{
    return uint16(
        (uint16(rgb.x) << 0) | (uint16(rgb.y) << 5) | (uint16(rgb.z) << 11));
}

HOST_DEVICE constexpr uint16 float3_to_rgb565(float3 rgb)
{
    const float r = clamp(rgb.x, 0.f, 1.f);
    const float g = clamp(rgb.y, 0.f, 1.f);
    const float b = clamp(rgb.z, 0.f, 1.f);
    return uint16(
        (uint16(r * 31.0f) << 0) | (uint16(g * 63.0f) << 5) | (uint16(b * 31.0f) << 11));
}

HOST_DEVICE constexpr float3 rgb565_to_float3(uint16 rgb)
{
    return make_vector3<float3>(
        float((rgb >> 0) & 0x1F) / 31.0f,
        float((rgb >> 5) & 0x3F) / 63.0f,
        float((rgb >> 11) & 0x1F) / 31.0f);
}

HOST_DEVICE constexpr uint32 float3_to_rgb888(float3 rgb)
{
    const float r = clamp(rgb.x, 0.f, 1.f);
    const float g = clamp(rgb.y, 0.f, 1.f);
    const float b = clamp(rgb.z, 0.f, 1.f);
    return (uint32(r * 255.0f) << 0) | (uint32(g * 255.0f) << 8) | (uint32(b * 255.0f) << 16) | 0xff000000;
}

HOST_DEVICE constexpr float3 rgb888_to_float3(uint32 rgb)
{
    return make_vector3<float3>(
        float((rgb >> 0) & 0xFF) / 255.0f,
        float((rgb >> 8) & 0xFF) / 255.0f,
        float((rgb >> 16) & 0xFF) / 255.0f);
}

HOST_DEVICE constexpr float3 rgb101210_to_float3(uint32 rgb)
{
    /*return make_vector3<float3>(
        float((rgb >> 0) & 0x3FF) / 1020.0f,
        float((rgb >> 10) & 0xFFF) / 4080.0f,
        float((rgb >> 22) & 0x3FF) / 1020.0f);*/

    return make_vector3<float3>(
        float((rgb >> 0) & 0x3FF) / 1023.0f,
        float((rgb >> 10) & 0xFFF) / 4095.0f,
        float((rgb >> 22) & 0x3FF) / 1023.0f);
}
HOST_DEVICE constexpr uint32 float3_to_rgb101210(float3 rgb)
{
    const float r = clamp(rgb.x, 0.f, 1.f);
    const float g = clamp(rgb.y, 0.f, 1.f);
    const float b = clamp(rgb.z, 0.f, 1.f);
    return // NOTE: was round here, but changed to a simple int cast as round is not constexpr
        (uint32(r * 1023.0f) << 0) | (uint32(g * 4095.0f) << 10) | (uint32(b * 1023.0f) << 22);
}

HOST_DEVICE constexpr uint32 rgb565_to_rgb888(uint16 rgb)
{
    return float3_to_rgb888(rgb565_to_float3(rgb));
}
HOST_DEVICE constexpr uint16 rgb888_to_rgb565(uint32 rgb)
{
    return float3_to_rgb565(rgb888_to_float3(rgb));
}
HOST_DEVICE float color_error(float3 a, float3 b)
{
    return length(a - b);
}
HOST_DEVICE constexpr float get_decimal_weight(uint8 weight, uint8 bitsPerWeight)
{
    return float(weight) / float((1 << bitsPerWeight) - 1);
}

HOST_DEVICE constexpr uint32 make_color_bits(float3 minColor, float3 maxColor)
{
    return float3_to_rgb565(minColor) | (uint32(float3_to_rgb565(maxColor)) << 16);
}

#if CFG_COLOR_SWAP_BYTE_ORDER
HOST_DEVICE uint32 swap_byte_order(uint32 aValue)
{
    return ((aValue & 0x000000ff) << 24)
        | ((aValue & 0x0000ff00) << 8)
        | ((aValue & 0x00ff0000) >> 8)
        | ((aValue & 0xff000000) >> 24);
}
HOST void swap_byte_order(uint32* aArray, std::size_t aCount)
{
    for (std::size_t i = 0; i < aCount; ++i)
        aArray[i] = swap_byte_order(aArray[i]);
}
HOST void swap_byte_order(StaticArray<uint32>& array)
{
    for (uint32& element : array) {
        element = swap_byte_order(element);
    }
}
#else
HOST_DEVICE uint32 swap_byte_order(uint32 aValue)
{
    return aValue;
}
HOST void swap_byte_order(uint32*, std::size_t) { }
HOST void swap_byte_order(StaticArray<uint32>&) { }
#endif // ~ CFG_COLOR_SWAP_BYTE_ORDER

HOST_DEVICE uint8 extract_bits(const uint32 bits, const StaticArray<uint32> array, const uint64 bitPtr)
{
    if (bits == 0)
        return uint8(0);

#if !CFG_COLOR_SWAP_BYTE_ORDER
    const uint32 ptrWord = cast<uint32>(bitPtr / 32);
    const uint32 ptrBit = cast<uint32>(bitPtr % 32);
    const uint32 bitsLeft = 32 - ptrBit;
    // Need to be careful not to try to shift >= 32 steps (undefined)
    const uint32 upperMask = (bitsLeft == 32) ? 0xFFFFFFFF : (~(0xFFFFFFFFu << bitsLeft));
    if (bitsLeft >= bits) {
        uint32 val = upperMask & array[ptrWord];
        val >>= (bitsLeft - bits);
        check(val < uint32(1 << bits));
        return uint8(val);
    } else {
        uint32 val = (upperMask & array[ptrWord]) << (bits - bitsLeft);
        val |= array[ptrWord + 1] >> (32 - (bits - bitsLeft));
        check(val < uint32(1 << bits));
        return uint8(val);
    }
#else
    auto const ptrWord = bitPtr / 8;

    uint16 dst;
    std::memcpy(&dst, reinterpret_cast<uint8 const*>(array.data()) + ptrWord, sizeof(uint16));
    dst = uint16((dst << 8) | (dst >> 8));

    auto const ptrBit = bitPtr % 8;
    auto const shift = 16 - bits - ptrBit;
    auto const mask = (1u << bits) - 1;
    return uint8((dst >> shift) & mask);
#endif
}

[[maybe_unused]] inline float3 rgb_to_xyz(float3 rgb)
{
    // COPIED FROM http://www.easyrgb.com/en/math.php
    // sR, sG and sB (Standard RGB) input range = 0 � 255
    // X, Y and Z output refer to a D65/2� standard illuminant.

    const auto transform = [](float& a) {
        if (a > 0.04045f)
            a = std::pow((a + 0.055f) / 1.055f, 2.4f);
        else
            a = a / 12.92f;
    };
    transform(rgb.x);
    transform(rgb.y);
    transform(rgb.z);
    rgb = rgb * 100.0f;

    const float X = rgb.x * 0.4124f + rgb.y * 0.3576f + rgb.z * 0.1805f;
    const float Y = rgb.x * 0.2126f + rgb.y * 0.7152f + rgb.z * 0.0722f;
    const float Z = rgb.x * 0.0193f + rgb.y * 0.1192f + rgb.z * 0.9505f;
    return make_float3(X, Y, Z);
}

[[maybe_unused]] inline float3 xyz_to_cielab(float3 xyz)
{
    // COPIED FROM http://www.easyrgb.com/en/math.php
    // Reference-X, Y and Z refer to specific illuminants and observers.
    // Common reference values are available below in this same page.

    // D65 White.
    const auto referenceWhite = make_float3(95.0489f, 100.0f, 108.8840f);
    xyz = xyz / referenceWhite;

    const auto transform = [](float& a) {
        if (a > 0.008856f)
            a = std::pow(a, 1.0f / 3.0f);
        else
            a = (7.787f * a) + (16.0f / 116.0f);
    };
    transform(xyz.x);
    transform(xyz.y);
    transform(xyz.z);

    const float L = (116.0f * xyz.y) - 16.0f;
    const float a = 500.0f * (xyz.x - xyz.y);
    const float b = 200.0f * (xyz.y - xyz.z);
    return make_float3(L, a, b);
}

[[maybe_unused]] inline float3 rgb_to_lab(const float3& rgb)
{
    return xyz_to_cielab(rgb_to_xyz(rgb));
}

[[maybe_unused]] inline float3 rgb_to_yuv(const float3& rgb)
{
    // https://en.wikipedia.org/wiki/YUV
    constexpr float w_r = 0.299f;
    constexpr float w_g = 0.587f;
    constexpr float w_b = 0.114f;
    constexpr float u_max = 0.436f;
    constexpr float v_max = 0.615f;

    const float y = w_r * rgb.x + w_g * rgb.y + w_b * rgb.z;
    const float u = u_max * (rgb.z - y) / (1.0f - w_b);
    const float v = v_max * (rgb.x - y) / (1.0f - w_r);
    return make_float3(y, u, v);
}

// Taken from "Moving Frostbite to PBR" page 88:
// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
HOST_DEVICE float3 approximationSRgbToLinear(float3 sRGBCol)
{
    return make_float3(std::pow(sRGBCol.x, 2.2f), std::pow(sRGBCol.y, 2.2f), std::pow(sRGBCol.z, 2.2f));
}
HOST_DEVICE float3 approximationLinearToSRGB(float3 linearCol)
{
    return make_float3(std::pow(linearCol.x, 1.0f / 2.2f), std::pow(linearCol.y, 1.0f / 2.2f), std::pow(linearCol.z, 1.0f / 2.2f));
}
HOST_DEVICE float3 accurateSRGBToLinear(float3 sRGBCol)
{
    const auto convertChannel = [](float sRGBCol) {
        const float linearRGBLo = sRGBCol / 12.92f;
        const float linearRGBHi = std::pow((sRGBCol + 0.055f) / 1.055f, 2.4f);
        const float linearRGB = (sRGBCol <= 0.04045f) ? linearRGBLo : linearRGBHi;
        return linearRGB;
    };
    return make_float3(convertChannel(sRGBCol.x), convertChannel(sRGBCol.y), convertChannel(sRGBCol.z));
}
HOST_DEVICE float3 accurateLinearToSRGB(float3 linearCol)
{
    const auto convertChannel = [](float linearCol) {
        const float sRGBLo = linearCol * 12.92f;
        const float sRGBHi = (std::pow(std::abs(linearCol), 1.0f / 2.4f) * 1.055f) - 0.055f;
        const float sRGB = (linearCol <= 0.0031308f) ? sRGBLo : sRGBHi;
        return sRGB;
    };
    return make_float3(convertChannel(linearCol.x), convertChannel(linearCol.y), convertChannel(linearCol.z));
}

} // namespace ColorUtils