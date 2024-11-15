#include "voxel_textures.h"
#include "color_utils.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include <charconv>
#include <filesystem>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <unordered_map>
//
#include "typedefs.h"

static const auto rootFolder = std::filesystem::path(ROOT_FOLDER);
// const auto textureFolder = rootFolder / "assets" / "textures";

static std::vector<std::string> stringSplit(std::string str, const std::string& delimiter)
{
    // https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
    std::vector<std::string> out;
    size_t pos = 0;
    while ((pos = str.find(delimiter)) != std::string::npos) {
        out.push_back(str.substr(0, pos));
        str.erase(0, pos + delimiter.length());
    }
    if (!str.empty())
        out.push_back(str);
    return out;
}

static std::string stringJoin(gsl::span<const std::string> parts, const std::string& separator)
{
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0)
            out += separator;
        out += parts[i];
    }
    return out;
}

static float3 computeAverageColor(const VoxelTexturesCPU& images, const auto convertToColorSpace)
{
    size_t count = 0;
    double3 sum = make_double3(0.0);
    for (const auto& image : std::array { images.all, images.side }) {
        for (const auto& pixel : image.pixels) {
            const float3 rgb = make_float3(pixel.r, pixel.g, pixel.b) / 255.0f;
            const float3 colorSpaced = convertToColorSpace(rgb);
            sum = sum + make_double3(colorSpaced);
        }
        count += image.pixels.size();
    }
    return make_float3(sum / (double)count / 255.0);
}

static std::string joinString(auto begin, auto end, std::string separator = "")
{
    std::string out = "";
    for (auto iter = begin; iter != end; ++iter) {
        if (iter != begin)
            out += separator;
        out += *iter;
    }
    return out;
}

VoxelTextures VoxelTextures::createFromStructuredFolder(const std::filesystem::path& textureFolder)
{
    VoxelTextures out {};
    for (const auto& entry : std::filesystem::directory_iterator(textureFolder)) {
        if (!std::filesystem::is_regular_file(entry))
            continue;

        const std::string delimiter = "_";
        const auto fileName = std::filesystem::path(entry).filename().string();
        const auto fileNameWithoutExtension = fileName.substr(0, fileName.find("."));
        const auto fileNameParts = stringSplit(fileNameWithoutExtension, delimiter);
        if (fileNameParts.size() < 3) {
            printf("Invalid texture file name \"%s\"; Must consist of at least 3 parts separated by \"%s\"\n", fileName.c_str(), delimiter.c_str());
            continue;
        }

        int idx = -1;
        std::from_chars(fileNameParts[0].data(), fileNameParts[0].data() + fileNameParts[0].size(), idx);
        if (idx < 0) {
            printf("Invalid texture index \"%i\" for file \"%s\"\n", idx, fileName.c_str());
            continue;
        }

        if (idx >= (int)out.cpuMaterials.size())
            out.cpuMaterials.resize(idx + 1);

        Image image { entry };
        const auto partStr = fileNameParts.back();
        auto& material = out.cpuMaterials[idx];
        if (partStr == "all") {
            material.all = image;
        } else if (partStr == "top") {
            material.top = image;
        } else if (partStr == "side") {
            material.side = image;
        } else {
            printf("Unsupported texture face \"%s\"\n", partStr.c_str());
            continue;
        }

        if (idx >= (int)out.materialNames.size())
            out.materialNames.resize(idx + 1);
        out.materialNames[idx] = joinString(std::begin(fileNameParts) + 1, std::end(fileNameParts) - 1, "_");
    }

    std::vector<VoxelTexturesGPU> gpuMaterials;
    int i = 0;
    for (auto& material : out.cpuMaterials) {
        material.averageRGB = computeAverageColor(material, [](float3 rgb) { return rgb; });
        material.averageYUV = computeAverageColor(material, [](float3 rgb) { return ColorUtils::rgb_to_yuv(rgb); });
        material.averageLAB = computeAverageColor(material, [](float3 rgb) { return ColorUtils::xyz_to_cielab(ColorUtils::rgb_to_xyz(rgb)); });
        printf("material %s has average color (%f, %f, %f)\n", out.materialNames[i++].c_str(), material.averageRGB.x, material.averageRGB.y, material.averageRGB.z);

        if (material.top.is_valid()) {
            checkAlways(material.side.is_valid());
        } else {
            checkAlways(material.all.is_valid());
            checkAlways(!material.side.is_valid());
        }

        VoxelTexturesGPU gpuMaterial;
        if (material.top.is_valid())
            gpuMaterial.top = material.top.createCudaTexture();
        if (material.side.is_valid())
            gpuMaterial.side = material.side.createCudaTexture();
        if (material.all.is_valid())
            gpuMaterial.all = material.all.createCudaTexture();
        gpuMaterials.push_back(gpuMaterial);
    }
    out.gpuMaterials = StaticArray<VoxelTexturesGPU>::allocate("VoxelTextures::gpuMaterials", gpuMaterials, EMemoryType::GPU_Malloc);
    return out;
}

static float computeColorError(const VoxelTexturesCPU& material, const float3& colorRGB, const float3& colorYUV, const float3& colorLAB)
{
    // const float3 diff = (material.averageLAB - colorLAB) * make_float3(0.3f, 1.0f, 1.0f);
    const float3 diff = material.averageLAB - colorLAB;
    return dot(diff, diff);
}

template <typename Colors>
VoxelTextures VoxelTextures::createSubsetFromFolder(const std::filesystem::path& textureFolder, const Colors& colors, size_t numColors)
{
    [[maybe_unused]] constexpr uint32_t MaxNumMaterials = MyGPUHashDAG<EMemoryType::CPU>::NumMaterials;

    std::unordered_map<std::string, VoxelTexturesCPU> materialsLUT;
    for (const auto& entry : std::filesystem::directory_iterator(textureFolder)) {
        if (!std::filesystem::is_regular_file(entry))
            continue;

        const std::string delimiter = "_";
        const auto fileName = std::filesystem::path(entry).filename().string();
        const auto fileNameWithoutExtension = fileName.substr(0, fileName.find("."));
        const auto fileNameParts = stringSplit(fileNameWithoutExtension, delimiter);
        Image inImage { entry };
        if (inImage.hasAlphaChannel)
            continue;
        checkAlways(!inImage.pixels.empty());

        Image* pOutImage;
        if (fileNameParts.back() == "top") {
            const auto materialName = stringJoin(gsl::span(fileNameParts).subspan(0, fileNameParts.size() - 1), "_");
            pOutImage = &materialsLUT[materialName].top;
        } else if (fileNameParts.back() == "side") {
            const auto materialName = stringJoin(gsl::span(fileNameParts).subspan(0, fileNameParts.size() - 1), "_");
            pOutImage = &materialsLUT[materialName].side;
        } else {
            const auto& materialName = fileNameWithoutExtension;
            pOutImage = &materialsLUT[materialName].all;
        }

        *pOutImage = inImage;
    }

    std::vector<std::string> materialNames;
    std::vector<VoxelTexturesCPU> materialsCPU;
    for (auto [name, material] : materialsLUT) {
        double3 rgbSum, yuvSum, labSum;
        int numPixels = 0;
        rgbSum = yuvSum = labSum = make_double3(0.0f);

        if (!material.all.is_valid() && (!material.side.is_valid() || !material.top.is_valid())) {
            printf("Missing textures for block type \"%s\"\n", name.c_str());
            continue;
        }

        const auto visitTexture = [&](const Image& image) {
            for (const auto& pixel : image.pixels) {
                const auto rgb = make_float3(pixel.b, pixel.g, pixel.r) / 255.0f;
                const auto yuv = ColorUtils::rgb_to_yuv(rgb);
                const auto lab = ColorUtils::rgb_to_lab(rgb);
                rgbSum = rgbSum + make_double3(rgb);
                yuvSum = yuvSum + make_double3(yuv);
                labSum = labSum + make_double3(lab);
                ++numPixels;
            }
        };
        visitTexture(material.all);
        visitTexture(material.top);
        visitTexture(material.side);

        checkAlways(material.all.is_valid() || (material.side.is_valid() && material.top.is_valid()));

        material.averageRGB = make_float3(rgbSum / (double)numPixels);
        material.averageYUV = make_float3(yuvSum / (double)numPixels);
        material.averageLAB = make_float3(labSum / (double)numPixels);

        materialNames.push_back(name);
        materialsCPU.push_back(material);
    }
    check(materialNames.size() == materialsCPU.size());

#if EDITS_ENABLE_MATERIALS
    struct ColorOccurrence {
        uint32_t colorRGB8;
        uint32_t numOccurrences;
    };
    std::vector<ColorOccurrence> colorsRGB8;
    {
        PROFILE_SCOPE("Count Color Occurrences");
        std::unordered_map<uint32_t, uint32_t> uniqueColorsRGB8;
        for (size_t colorIdx = 0; colorIdx < numColors; ++colorIdx) {
            if constexpr (std::is_same_v<Colors, BasicDAGUncompressedColors>) {
                const uint32_t colorRGB8 = colors.get_default_leaf().colors[colorIdx];
                uniqueColorsRGB8[colorRGB8]++;
            } else {
                const auto& colorLeaf = colors.get_default_leaf();
                const uint32_t colorRGB8 = ColorUtils::float3_to_rgb888(colorLeaf.get_color(colorIdx).get_color());
                uniqueColorsRGB8[colorRGB8]++;
            }
        }
        for (const auto [key, value] : uniqueColorsRGB8) {
            colorsRGB8.push_back({ .colorRGB8 = key, .numOccurrences = value });
        }
    }

    while (materialsCPU.size() > MaxNumMaterials) {
        printf("Reducing number of voxel textures %zu => %u\n", materialsCPU.size(), MaxNumMaterials);
        // Greedy algorithm that picks the materials that lead to the lowest image error.
        std::vector<double> costOfRemovingMaterial(materialsCPU.size(), 0.0);
        std::vector<size_t> numColorsPerMaterial(materialsCPU.size(), 0);
        for (const auto& [colorRGB8, numOccurences] : colorsRGB8) {
            const float3 rgb = ColorUtils::rgb888_to_float3(colorRGB8);
            const float3 yuv = ColorUtils::rgb_to_yuv(rgb);
            const float3 lab = ColorUtils::rgb_to_lab(rgb);

            uint8_t optimalMaterial, secondMostOptimalMaterial;
            optimalMaterial = secondMostOptimalMaterial = 0xFF;
            float smallestError, secondSmallestError;
            smallestError = secondSmallestError = std::numeric_limits<float>::max();
            for (size_t materialIdx = 0; materialIdx < materialsCPU.size(); ++materialIdx) {
                const auto& material = materialsCPU[materialIdx];
                // const float3 difference = (material.averageLAB - lab) * make_float3(0.3f, 1.0f, 1.0f);
                // const float summedSquaredDifference = (difference.x * difference.x) + (difference.y * difference.y) + (difference.z * difference.z);
                const float colorError = computeColorError(material, rgb, yuv, lab);

                if (colorError < smallestError) {
                    // New optimal material; Old optimal material becomes becomes second best.
                    secondMostOptimalMaterial = optimalMaterial;
                    secondSmallestError = smallestError;

                    optimalMaterial = (uint8_t)materialIdx;
                    smallestError = colorError;
                } else if (colorError < secondSmallestError) {
                    secondMostOptimalMaterial = (uint8_t)materialIdx;
                    secondSmallestError = colorError;
                }
            }
            check(smallestError <= secondSmallestError);
            costOfRemovingMaterial[optimalMaterial] += numOccurences * double(secondSmallestError - smallestError);
            numColorsPerMaterial[optimalMaterial] += numOccurences;
        }

        // Select the material with the lowest average error.
        // std::transform(std::begin(costOfRemovingMaterial), std::end(costOfRemovingMaterial), std::begin(numColorsPerMaterial), std::begin(costOfRemovingMaterial),
        //    [](double summedCostOfRemovingMaterial, size_t numColorsThisMaterial) { return summedCostOfRemovingMaterial / std::log((double)numColorsThisMaterial); });
        const auto leastPopularMaterial = std::distance(
            std::begin(costOfRemovingMaterial),
            std::min_element(std::begin(costOfRemovingMaterial), std::end(costOfRemovingMaterial)));
        materialsCPU.erase(std::begin(materialsCPU) + leastPopularMaterial);
        materialNames.erase(std::begin(materialNames) + leastPopularMaterial);
    }
#else
    materialNames.resize(MaxNumMaterials);
    materialsCPU.resize(MaxNumMaterials);
#endif
    check(materialNames.size() == materialsCPU.size());
    check(materialNames.size() <= MaxNumMaterials);

    printf("Material names:\n");
    for (const auto& materialName : materialNames) {
        printf("%s\n", materialName.c_str());
    }

    std::vector<VoxelTexturesGPU> gpuMaterials;
    for (const auto& material : materialsCPU) {
        VoxelTexturesGPU gpuMaterial;
        if (material.top.is_valid())
            gpuMaterial.top = material.top.createCudaTexture();
        if (material.side.is_valid())
            gpuMaterial.side = material.side.createCudaTexture();
        if (material.all.is_valid())
            gpuMaterial.all = material.all.createCudaTexture();
        gpuMaterials.push_back(gpuMaterial);
    }

    return VoxelTextures {
        .materialNames = std::move(materialNames),
        .cpuMaterials = std::move(materialsCPU),
        .gpuMaterials = StaticArray<VoxelTexturesGPU>::allocate("VoxelTextures::gpuMaterials", gpuMaterials, EMemoryType::GPU_Malloc)
    };
}

void VoxelTextures::free()
{
    for (auto& material : gpuMaterials.copy_to_cpu()) {
        const auto tryFree = [](auto mat) {
            if (mat.cuArray)
                mat.free();
        };
        tryFree(material.all);
        tryFree(material.side);
        tryFree(material.top);
    }
    gpuMaterials.free();
}

uint32_t VoxelTextures::getClosestMaterial(const float3& rgb) const
{
    const float3 yuv = ColorUtils::rgb_to_yuv(rgb);
    const float3 lab = ColorUtils::xyz_to_cielab(ColorUtils::rgb_to_xyz(rgb));

    uint32_t closestMaterialIdx = (uint32_t)-1;
    float closest = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < cpuMaterials.size(); ++i) {
        // const auto distance = length_squared(colorYUV - cpuMaterials[i].averageYUV);
        const auto colorError = computeColorError(cpuMaterials[i], rgb, yuv, lab);
        if (colorError < closest) {
            closestMaterialIdx = i;
            closest = colorError;
        }
    }
    return closestMaterialIdx;
}

void VoxelTexturesCPU::writeTo(BinaryWriter& writer) const
{
    writer.write(all);
    writer.write(top);
    writer.write(side);
    writer.write(averageRGB);
    writer.write(averageYUV);
    writer.write(averageLAB);
}

void VoxelTexturesCPU::readFrom(BinaryReader& reader)
{
    reader.read(all);
    reader.read(top);
    reader.read(side);
    reader.read(averageRGB);
    reader.read(averageYUV);
    reader.read(averageLAB);
}

void VoxelTextures::writeTo(BinaryWriter& writer) const
{
    writer.write(materialNames);
    writer.write(cpuMaterials);
}

void VoxelTextures::readFrom(BinaryReader& reader)
{
    reader.read(materialNames);
    reader.read(cpuMaterials);
    checkAlways(!gpuMaterials.is_valid());

    std::vector<VoxelTexturesGPU> localGpuMaterials;
    for (const auto& material : cpuMaterials) {
        VoxelTexturesGPU gpuMaterial;
        if (material.top.is_valid())
            gpuMaterial.top = material.top.createCudaTexture();
        if (material.side.is_valid())
            gpuMaterial.side = material.side.createCudaTexture();
        if (material.all.is_valid())
            gpuMaterial.all = material.all.createCudaTexture();
        localGpuMaterials.push_back(gpuMaterial);
    }
    gpuMaterials = StaticArray<VoxelTexturesGPU>::allocate("VoxelTextures::gpuMaterials", localGpuMaterials, EMemoryType::GPU_Malloc);
}

template VoxelTextures VoxelTextures::createSubsetFromFolder(const std::filesystem::path&, const BasicDAGCompressedColors&, size_t);
template VoxelTextures VoxelTextures::createSubsetFromFolder(const std::filesystem::path&, const BasicDAGUncompressedColors&, size_t);
