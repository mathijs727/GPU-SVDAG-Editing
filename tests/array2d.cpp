#include "array2d.h"
#include "depth_image.h"
#include "image.h"
#include <algorithm>
#include <bit>
#include <catch2/catch_all.hpp>

TEST_CASE("array2d")
{
    auto array2D = StaticArray2D<int>::allocate("array2D", 8, 8, EMemoryType::CPU);
    REQUIRE(array2D.width == 8);
    REQUIRE(array2D.height == 8);
    REQUIRE(array2D.buffer.size() == 64);
    array2D.free();
}

TEST_CASE("DepthRangeImage")
{
    DepthRangeImage<EDepthTest::LessThan> depthImage { 8, 8 };
    REQUIRE(depthImage.width == 8);
    REQUIRE(depthImage.height == 8);

    depthImage.addDepth(3, 5, 124);
    depthImage.addDepth(1, 0, 125);
    depthImage.addDepth(0, 2, 126);

    depthImage.generateMipChain();

    for (uint32_t y = 0; y < 8; ++y) {
        for (uint32_t x = 0; x < 8; ++x) {
            REQUIRE(depthImage.testDepthApprox(x, x + 1, y, y + 1, 123) == true);
        }
    }
    for (uint32_t y = 0; y < 8; y += 2) {
        for (uint32_t x = 0; x < 8; x += 2) {
            REQUIRE(depthImage.testDepthApprox(x, x + 2, y, y + 2, 123) == true);
        }
    }

    // TEST 2x2 regions
    REQUIRE(depthImage.testDepthApprox(0, 2, 0, 2, 124) == true);
    REQUIRE(depthImage.testDepthApprox(0, 2, 0, 2, 125) == false);
    REQUIRE(depthImage.testDepthApprox(2, 4, 0, 2, 125) == true);
    REQUIRE(depthImage.testDepthApprox(4, 6, 0, 2, 125) == true);
    REQUIRE(depthImage.testDepthApprox(6, 8, 0, 2, 125) == true);

    REQUIRE(depthImage.testDepthApprox(0, 2, 2, 4, 125) == true);
    REQUIRE(depthImage.testDepthApprox(0, 2, 2, 4, 126) == false);
    REQUIRE(depthImage.testDepthApprox(2, 4, 2, 4, 125) == true);
    REQUIRE(depthImage.testDepthApprox(4, 6, 2, 4, 125) == true);
    REQUIRE(depthImage.testDepthApprox(6, 8, 2, 4, 125) == true);

    REQUIRE(depthImage.testDepthApprox(0, 2, 4, 6, 125) == true);
    REQUIRE(depthImage.testDepthApprox(2, 4, 4, 6, 125) == false);
    REQUIRE(depthImage.testDepthApprox(2, 4, 4, 6, 124) == false);
    REQUIRE(depthImage.testDepthApprox(2, 4, 4, 6, 123) == true);
    REQUIRE(depthImage.testDepthApprox(4, 6, 4, 6, 125) == true);
    REQUIRE(depthImage.testDepthApprox(6, 8, 4, 6, 125) == true);

    REQUIRE(depthImage.testDepthApprox(0, 2, 6, 8, 125) == true);
    REQUIRE(depthImage.testDepthApprox(2, 4, 6, 8, 125) == true);
    REQUIRE(depthImage.testDepthApprox(4, 6, 6, 8, 125) == true);
    REQUIRE(depthImage.testDepthApprox(6, 8, 6, 8, 125) == true);

    // TEST LARGER REGIONS
    REQUIRE(depthImage.testDepthApprox(2, 5, 0, 4, 128) == true);

    REQUIRE(depthImage.testDepthApprox(0, 5, 0, 4, 127) == false);
    REQUIRE(depthImage.testDepthApprox(0, 5, 0, 4, 126) == false);
    REQUIRE(depthImage.testDepthApprox(0, 5, 0, 4, 125) == false);
    REQUIRE(depthImage.testDepthApprox(0, 5, 0, 4, 124) == true);

    REQUIRE(depthImage.testDepthApprox(2, 4, 4, 6, 125) == false);
    REQUIRE(depthImage.testDepthApprox(2, 4, 4, 6, 124) == false);
    REQUIRE(depthImage.testDepthApprox(2, 4, 4, 6, 123) == true);
}
