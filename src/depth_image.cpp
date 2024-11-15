#include "depth_image.h"
#include <algorithm>
#include <bit>

template <EDepthTest depthTest>
DepthRangeImage<depthTest>::DepthRangeImage(uint32_t width, uint32_t height)
    : width(width)
    , height(height)
{
    mipChain.emplace_back((size_t)(width * height));
    if constexpr (depthTest == EDepthTest::LessThan) {
        std::fill(std::begin(mipChain[0]), std::end(mipChain[0]), std::numeric_limits<uint32_t>::max());
    } else {
        std::fill(std::begin(mipChain[0]), std::end(mipChain[0]), 0);
    }
}

template <EDepthTest depthTest>
void DepthRangeImage<depthTest>::generateMipChain()
{
    uint32_t mipWidth = width, mipHeight = height;
    while (mipWidth > 1 && mipHeight > 1) {
        const uint32_t parentHeight = mipHeight;

        mipWidth >>= 1;
        mipHeight >>= 1;
        mipChain.emplace_back((size_t)(mipWidth * mipHeight));
        auto& parentImage = mipChain[mipChain.size() - 2]; // AFTER emplace_back() which may invalidate pointers.
        auto& mipImage = mipChain.back();

        for (uint32_t y = 0; y < mipHeight; ++y) {
            for (uint32_t x = 0; x < mipWidth; ++x) {
                uint32_t x2 = x << 1;
                uint32_t y2 = y << 1;
                if constexpr (depthTest == EDepthTest::LessThan) {
                    mipImage[y * mipWidth + x] = std::min(
                        std::min(
                            parentImage[(y2 + 0) * parentHeight + (x2 + 0)],
                            parentImage[(y2 + 0) * parentHeight + (x2 + 1)]),
                        std::min(
                            parentImage[(y2 + 1) * parentHeight + (x2 + 0)],
                            parentImage[(y2 + 1) * parentHeight + (x2 + 1)]));
                } else {
                    mipImage[y * mipWidth + x] = std::max(
                        std::max(
                            parentImage[(y2 + 0) * parentHeight + (x2 + 0)],
                            parentImage[(y2 + 0) * parentHeight + (x2 + 1)]),
                        std::max(
                            parentImage[(y2 + 1) * parentHeight + (x2 + 0)],
                            parentImage[(y2 + 1) * parentHeight + (x2 + 1)]));
                }
            }
        }
    }
}

template <EDepthTest depthTest>
bool DepthRangeImage<depthTest>::testDepthApprox(uint32_t minX, uint32_t maxX, uint32_t minY, uint32_t maxY, uint32_t value) const
{
    const uint32_t extentX = maxX - minX;
    const uint32_t extentY = maxY - minY;
    const uint32_t smallestExtent = std::min(extentX, extentY);
    const uint32_t mostSignificantBit = 32 - std::countl_zero(smallestExtent);
    const uint32_t mipMapLevel = std::max(mostSignificantBit, 2u) - 2u;

    const uint32_t mipWidth = width >> mipMapLevel;
    const uint32_t mipHeight = height >> mipMapLevel;
    const auto& mipImage = mipChain[mipMapLevel];

    minX >>= mipMapLevel; // Round down
    maxX = ((maxX - 1) >> mipMapLevel) + 1; // Round up
    minY >>= mipMapLevel;
    maxY = ((maxY - 1) >> mipMapLevel) + 1;

    minX = std::clamp(minX, 0u, mipWidth);
    maxX = std::clamp(maxX, 0u, mipWidth);
    minY = std::clamp(minY, 0u, mipHeight);
    maxY = std::clamp(maxY, 0u, mipHeight);

    for (uint32_t y = minY; y < maxY; ++y) {
        for (uint32_t x = minX; x < maxX; ++x) {
            const uint32_t threshold = mipImage[y * mipWidth + x];
            if constexpr (depthTest == EDepthTest::LessThan) {
                if (value >= threshold)
                    return false;
            } else {
                if (value <= threshold)
                    return false;
            }
        }
    }
    return true;
}

template struct DepthRangeImage<EDepthTest::LessThan>;
template struct DepthRangeImage<EDepthTest::GreaterThan>;