#pragma once
#include <filesystem>
#include <vector>

struct FreeImagePixel {
    uint8_t r, g, b, a;
};

void load_image(const std::filesystem::path& filePath, std::vector<FreeImagePixel>& pixels,
    int& width, int& height, bool& hasAlphaChannel);