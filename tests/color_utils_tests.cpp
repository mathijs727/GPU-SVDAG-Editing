#include <catch2/catch_all.hpp>
// ^^^ INCLUDE FIRST ^^^
#include "color_utils.h"
#include "cuda_math.h" // operator<<

constexpr float margin = 0.01f;

TEST_CASE("RGB to XYZ", "[ColorUtils]")
{
    // Reference solutions from:
    // http://colormine.org/convert/rgb-to-xyz
    SECTION("White")
    {
        const float3 rgb888 = make_float3(1.0f);
        const float3 cieXYZ = ColorUtils::rgb_to_xyz(rgb888);

        CAPTURE(rgb888, cieXYZ);
        REQUIRE(cieXYZ.x == Catch::Approx(95.05f).margin(margin));
        REQUIRE(cieXYZ.y == Catch::Approx(100.0f).margin(margin));
        REQUIRE(cieXYZ.z == Catch::Approx(108.9f).margin(margin));
    }

    SECTION("Red")
    {
        const float3 rgb888 = make_float3(1.0f, 0.0f, 0.0f);
        const float3 cieXYZ = ColorUtils::rgb_to_xyz(rgb888);

        CAPTURE(rgb888, cieXYZ);
        REQUIRE(cieXYZ.x == Catch::Approx(41.24f).margin(margin));
        REQUIRE(cieXYZ.y == Catch::Approx(21.26f).margin(margin));
        REQUIRE(cieXYZ.z == Catch::Approx(1.93f).margin(margin));
    }

    SECTION("Green")
    {
        const float3 rgb888 = make_float3(0.0f, 1.0f, 0.0f);
        const float3 cieXYZ = ColorUtils::rgb_to_xyz(rgb888);

        CAPTURE(rgb888, cieXYZ);
        REQUIRE(cieXYZ.x == Catch::Approx(35.76f).margin(margin));
        REQUIRE(cieXYZ.y == Catch::Approx(71.52f).margin(margin));
        REQUIRE(cieXYZ.z == Catch::Approx(11.92f).margin(margin));
    }

    SECTION("Blue")
    {
        const float3 rgb888 = make_float3(0.0f, 0.0f, 1.0f);
        const float3 cieXYZ = ColorUtils::rgb_to_xyz(rgb888);

        CAPTURE(rgb888, cieXYZ);
        REQUIRE(cieXYZ.x == Catch::Approx(18.05f).margin(margin));
        REQUIRE(cieXYZ.y == Catch::Approx(7.22f).margin(margin));
        REQUIRE(cieXYZ.z == Catch::Approx(95.05f).margin(margin));
    }
}

TEST_CASE("RGB to LAB", "[ColorUtils]")
{
    // Reference solutions from:
    // http://colormine.org/convert/rgb-to-lab
    SECTION("White")
    {
        const float3 rgb888 = make_float3(1.0f, 1.0f, 1.0f);
        const float3 cieLAB = ColorUtils::xyz_to_cielab(ColorUtils::rgb_to_xyz(rgb888));

        CAPTURE(rgb888, cieLAB);
        REQUIRE(cieLAB.x == Catch::Approx(100.0f).margin(margin));
        REQUIRE(cieLAB.y == Catch::Approx(0.00526049995830391f).margin(margin));
        REQUIRE(cieLAB.z == Catch::Approx(-0.010408184525267927f).margin(margin));
    }

    SECTION("Red")
    {
        const float3 rgb888 = make_float3(1.0f, 0.0f, 0.0f);
        const float3 cieLAB = ColorUtils::xyz_to_cielab(ColorUtils::rgb_to_xyz(rgb888));

        CAPTURE(rgb888, cieLAB);
        REQUIRE(cieLAB.x == Catch::Approx(53.23288178584245f).margin(margin));
        REQUIRE(cieLAB.y == Catch::Approx(80.10930952982204f).margin(margin));
        REQUIRE(cieLAB.z == Catch::Approx(67.22006831026425).margin(margin));
    }

    SECTION("Green")
    {
        const float3 rgb888 = make_float3(0.0f, 1.0f, 0.0f);
        const float3 cieLAB = ColorUtils::xyz_to_cielab(ColorUtils::rgb_to_xyz(rgb888));

        CAPTURE(rgb888, cieLAB);
        REQUIRE(cieLAB.x == Catch::Approx(87.73703347354422f).margin(margin));
        REQUIRE(cieLAB.y == Catch::Approx(-86.18463649762525f).margin(margin));
        REQUIRE(cieLAB.z == Catch::Approx(83.18116474777854f).margin(margin));
    }

    SECTION("Blue")
    {
        const float3 rgb888 = make_float3(0.0f, 0.0f, 1.0f);
        const float3 cieLAB = ColorUtils::xyz_to_cielab(ColorUtils::rgb_to_xyz(rgb888));

        CAPTURE(rgb888, cieLAB);
        REQUIRE(cieLAB.x == Catch::Approx(32.302586667249486f).margin(margin));
        REQUIRE(cieLAB.y == Catch::Approx(79.19666178930935f).margin(margin));
        REQUIRE(cieLAB.z == Catch::Approx(-107.86368104495168f).margin(margin));
    }
}
