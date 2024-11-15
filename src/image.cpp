#include "image.h"
#include "free_image_wrapper.h"

Image::Image(const std::filesystem::path& filePath)
{
    std::vector<FreeImagePixel> freeImagePixels;
    load_image(filePath, freeImagePixels, width, height, hasAlphaChannel);

    for (const FreeImagePixel& p : freeImagePixels) {
        pixels.push_back({ .r = p.r, .g = p.g, .b = p.b, .a = p.a });
    }
    fWidth = (float)width;
    fHeight = (float)height;
}