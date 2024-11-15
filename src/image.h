#pragma once
#include "binary_reader.h"
#include "binary_writer.h"
#include "cuda_error_check.h"
#include "cuda_math.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>

struct CUDATexture {
    cudaArray* cuArray { nullptr };
    cudaTextureObject_t cuTexture;

    void free()
    {
        cudaDestroyTextureObject(cuTexture);
        cudaFreeArray(cuArray);
    }
};

struct Image {
    struct Pixel {
        uint8_t r, g, b, a;

        void writeTo(BinaryWriter& writer) const
        {
            writer.write(r);
            writer.write(g);
            writer.write(b);
            writer.write(a);
        }
        void readFrom(BinaryReader& reader)
        {
            reader.read(r);
            reader.read(g);
            reader.read(b);
            reader.read(a);
        }
    };
    std::vector<Pixel> pixels;
    int width = 0, height = 0;
    float fWidth, fHeight;
    bool hasAlphaChannel;

public:
    Image() = default;
    Image(const std::filesystem::path& image);

    void writeTo(BinaryWriter& writer) const
    {
        writer.write(pixels);
        writer.write(width);
        writer.write(height);
        writer.write(fWidth);
        writer.write(fHeight);
        writer.write(hasAlphaChannel);
    }
    void readFrom(BinaryReader& reader)
    {
        reader.read(pixels);
        reader.read(width);
        reader.read(height);
        reader.read(fWidth);
        reader.read(fHeight);
        reader.read(hasAlphaChannel);
    }

    inline bool is_valid() const { return width > 0 && height > 0; }

    inline Pixel& a(int x, int y)
    {
        assert(x < width);
        assert(y < height);
        return pixels[y * width + x];
    }
    inline Pixel a(int x, int y) const
    {
        assert(x < width);
        assert(y < height);
        return pixels[y * width + x];
    }

    inline Pixel getWrapped(const int2& pixel) const noexcept
    {
        int2 wrappedPixel { pixel.x % width, pixel.y % height };
        if (wrappedPixel.x < 0)
            wrappedPixel.x += width;
        if (wrappedPixel.y < 0)
            wrappedPixel.y += height;
        return this->a(wrappedPixel.x, wrappedPixel.y);
    }
    inline Pixel sampleNN(const float2& texCoord) const noexcept
    {
        const int2 texel = ::truncateSigned(texCoord * make_float2(fWidth, fHeight));
        return getWrapped(texel);
    }
    inline Pixel sampleBilinear(const float2& texCoord) const noexcept
    {
        // May be negative and/or greater than resolution (both will use texture wrapping).O
        const float2 texel = texCoord * make_float2(fWidth, fHeight) - 0.5f;
        float2 integralPart;
        float2 fractionalPart { std::modf(texel.x, &integralPart.x), std::modf(texel.y, &integralPart.y) };
        if (fractionalPart.x < 0.0f)
            fractionalPart.x += 1.0f;
        if (fractionalPart.y < 0.0f)
            fractionalPart.y += 1.0f;

        const auto pixelLerp = [](const Pixel& lhs, const Pixel& rhs, float a) {
            const auto invA = 1.0f - a;
            return Pixel {
                (uint8_t)(invA * lhs.r + a * rhs.r),
                (uint8_t)(invA * lhs.g + a * rhs.g),
                (uint8_t)(invA * lhs.b + a * rhs.b),
                (uint8_t)(invA * lhs.a + a * rhs.a),
            };
        };
        const int2 iTexel = ::truncateSigned(integralPart);
        const auto NW = getWrapped(iTexel + make_int2(0, 0));
        const auto NE = getWrapped(iTexel + make_int2(1, 0));
        const auto SW = getWrapped(iTexel + make_int2(0, 1));
        const auto SE = getWrapped(iTexel + make_int2(1, 1));

        const auto north = pixelLerp(NW, NE, fractionalPart.x);
        const auto south = pixelLerp(SW, SE, fractionalPart.x);
        return pixelLerp(north, south, fractionalPart.y);
    }

    inline CUDATexture createCudaTexture() const
    {
        // FreeImage and CUDA don't agree on pixel layout (RGB vs BGR).
        std::vector<Pixel> flippedPixels(pixels.size());
        std::transform(std::begin(pixels), std::end(pixels), std::begin(flippedPixels),
            [](const Pixel& in) {
                return Pixel { in.b, in.g, in.r, in.a };
            });

        CUDATexture out;
        const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECKED_CALL cudaMallocArray(&out.cuArray, &channelDesc, width, height);
        cudaMemcpy2DToArray(out.cuArray, 0, 0, flippedPixels.data(), width * sizeof(Pixel), width * sizeof(Pixel), height, cudaMemcpyHostToDevice);

        cudaResourceDesc resourceDesc {};
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = out.cuArray;

        cudaTextureDesc textureDesc {};
        textureDesc.addressMode[0] = textureDesc.addressMode[1] = textureDesc.addressMode[2] = cudaTextureAddressMode::cudaAddressModeClamp;
        textureDesc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
        textureDesc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
        textureDesc.sRGB = false;
        textureDesc.normalizedCoords = true;
        textureDesc.maxAnisotropy = 0;
        textureDesc.mipmapFilterMode = cudaTextureFilterMode::cudaFilterModePoint;
        textureDesc.mipmapLevelBias = 0.0f;
        textureDesc.minMipmapLevelClamp = 0.0f;
        textureDesc.maxMipmapLevelClamp = 0.0f;

        CUDA_CHECKED_CALL cudaCreateTextureObject(&out.cuTexture, &resourceDesc, &textureDesc, nullptr);

        return out;
    }
};