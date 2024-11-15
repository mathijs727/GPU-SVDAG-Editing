#pragma once
#include "typedefs.h"
#include <algorithm>
#include <vector>

enum class EDepthTest {
    LessThan,
    GreaterThan
};
template <EDepthTest depthTest>
struct DepthRangeImage {
    std::vector<std::vector<uint32_t>> mipChain;
    uint32_t width, height;

    DepthRangeImage(uint32_t width, uint32_t height);

    inline void addDepth(uint32_t x, uint32_t y, uint32_t value)
    {
        check(x < width);
        check(y < height);
        auto& pixel = mipChain[0][y * width + x];
        if constexpr (depthTest == EDepthTest::LessThan)
            pixel = std::min(pixel, value);
        else
            pixel = std::max(pixel, value);
    }
    void generateMipChain();

    inline bool testDepth(uint32_t x, uint32_t y, uint32_t value) const
    {
        if constexpr (depthTest == EDepthTest::LessThan)
            return value < mipChain[0][y * width + x];
        else
            return value > mipChain[0][y * width + x];
    }

    // minX/minY are inclusive, maxX/maxY are exclusive.
    bool testDepthApprox(uint32_t minX, uint32_t maxX, uint32_t minY, uint32_t maxY, uint32_t value) const;
};
