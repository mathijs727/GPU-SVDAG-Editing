#ifdef _WIN32
#define NOMINMAX 1
#include <windows.h>
#endif
#define FREEIMAGE_LIB
#define FREEIMAGE_COLORORDER FREEIMAGE_COLORORDER_RGB
#include <FreeImage.h>
#include <FreeImagePlus.h>
// ^^^ Windows doesn't like it if we don't include this as the first file ^^^
#include "free_image_wrapper.h"
#include <cassert>

void load_image(
    const std::filesystem::path& filePath,
    std::vector<FreeImagePixel>& pixels, int& width, int& height, bool& hasAlphaChannel)
{
    assert(std::filesystem::exists(filePath));

    // Load image from disk.
    const std::string filePathString = filePath.string();
    fipImage image;
    [[maybe_unused]] const bool loadSuccess = image.load(filePathString.c_str());
    assert(image.isValid());
    assert(loadSuccess);
    width = image.getWidth();
    height = image.getHeight();
    hasAlphaChannel = image.isTransparent();

    // Convert image to RGBA8 format.
    [[maybe_unused]] const bool conversionSuccess = image.convertTo32Bits();
    assert(conversionSuccess);

    for (int y = 0; y != height; y++) {
        for (int x = 0; x != width; x++) {
            RGBQUAD rgb;
            image.getPixelColor(x, height - 1 - y, &rgb);
            pixels.push_back({ .r = rgb.rgbBlue, .g = rgb.rgbGreen, .b = rgb.rgbRed, .a = 255 });
        }
    }
}