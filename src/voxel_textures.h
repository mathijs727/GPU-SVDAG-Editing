#pragma once
#include "array.h"
#include "binary_reader.h"
#include "binary_writer.h"
#include "image.h"
#include <filesystem>
#include <vector>

struct VoxelTexturesCPU {
    Image all;
    Image top;
    Image side;

    float3 averageRGB;
    float3 averageYUV;
    float3 averageLAB;

    void writeTo(BinaryWriter& writer) const;
    void readFrom(BinaryReader& reader);
};
struct VoxelTexturesGPU {
    CUDATexture all;
    CUDATexture top;
    CUDATexture side;
};

struct VoxelTextures {
    std::vector<std::string> materialNames;
    std::vector<VoxelTexturesCPU> cpuMaterials;
    StaticArray<VoxelTexturesGPU> gpuMaterials;

    static VoxelTextures createFromStructuredFolder(const std::filesystem::path& folderPath);
    template <typename Colors>
    static VoxelTextures createSubsetFromFolder(const std::filesystem::path& folderPath, const Colors& colorsRGB, size_t numColors);
    void free();

    uint32_t getClosestMaterial(const float3& colorRGB) const;

    void writeTo(BinaryWriter& writer) const;
    void readFrom(BinaryReader& reader);
};