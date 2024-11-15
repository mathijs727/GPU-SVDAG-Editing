#pragma once
#include "array2d.h"
#include "cuda_math.h"
#include "image.h"
#include "path.h"
#include "typedefs.h"
#include "voxel_textures.h"

enum class EDebugColors : int {
    None,
    Index,
    Position,
    ColorTree,
    ColorBits,
    MinColor,
    MaxColor,
    Weight
};

constexpr uint32 CNumDebugColors = 8;

enum class ETool : int {
    Sphere,
    SpherePaint,
    SphereNoise,
    Cube,
    CubeCopy,
    CubeFill
};

constexpr uint32 CNumTools = 9;

struct ToolPath {
    uint3 centerPath;
    uint3 neighbourPath;
};

struct ToolInfo {
    ETool tool;
    ToolPath position;
    float radius;
    Path copySource = Path(0, 0, 0);
    Path copyDest = Path(0, 0, 0);

    ToolInfo() = default;
    ToolInfo(ETool tool, ToolPath position, float radius, uint3 copySource, uint3 copyDest)
        : tool(tool)
        , position(position)
        , radius(radius)
        , copySource(copySource)
        , copyDest(copyDest)
    {
    }

    HOST_DEVICE float3 addToolColor(const Path& path, float3 pixelColor) const
    {
        const auto addColor = [&](float toolStrength, float3 toolColor = make_float3(1, 0, 0)) {
            pixelColor = lerp(pixelColor, toolColor, clamp(100.0f * toolStrength, 0.0f, 0.5f));
        };

        switch (tool) {
        case ETool::Sphere:
        case ETool::SpherePaint:
        case ETool::SphereNoise: {
            addColor(sphere_strength(position.centerPath, path, radius));
        } break;
        case ETool::Cube:
        case ETool::CubeFill: {
            addColor(cube_strength(position.centerPath, path, radius));
        } break;
        case ETool::CubeCopy: {
            addColor(cube_strength(position.centerPath, path, radius));
            addColor(sphere_strength(copySource, path, 3), make_float3(0, 1, 0));
            addColor(sphere_strength(copyDest, path, 3), make_float3(0, 0, 1));
        } break;
        default:
            break;
        };

        return pixelColor;
    }

private:
    HOST_DEVICE static float cube_strength(const Path pos, const Path path, float radius)
    {
        return 1 - max(abs(pos.as_position() - path.as_position())) / radius;
    }
    HOST_DEVICE static float sphere_strength(const Path pos, const Path path, float radius)
    {
        return 1 - length(pos.as_position() - path.as_position()) / radius;
    }
};

namespace Tracer {
struct SurfaceInteraction {
    Path path;
    uint8_t materialId;

    float3 position;
    float3 normal;
    float3 dpdu;
    float3 dpdv;

    uint32_t diffuseColorSRGB8;

    HOST_DEVICE float3 transformDirectionToWorld(float3 local) const
    {
        return local.x * dpdu + local.y * dpdv + local.z * normal;
    }
};

struct TracePathsParams {
    // In
    double3 cameraPosition;
    double3 rayMin;
    double3 rayDDx;
    double3 rayDDy;

    // Out
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;
};

template <typename TDAG>
__global__ void trace_paths(TracePathsParams traceParams, const TDAG dag);

struct ColorParams {
    StaticArray<VoxelTexturesGPU> materialTextures;
    EDebugColors debugColors;
    uint32 debugColorsIndexLevel;
};
struct TraceColorsParams {
    // In
    ColorParams colorParams;

    // In/Out
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;
};

template <typename TDAG, typename TDAGColors>
__global__ void trace_colors(TraceColorsParams traceParams, const TDAG dag, const TDAGColors colors);

struct TraceShadowsParams {
    // In
    float shadowBias;
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;

    // In/Out
    StaticArray2D<float> sunLightSurface;
};

template <typename TDAG>
__global__ void trace_shadows(TraceShadowsParams params, const TDAG dag);

struct TraceAmbientOcclusionParams {
    // In
    uint32_t numSamples;
    float shadowBias;
    float aoRayLength;
    uint64_t randomSeed;
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;

    // Out
    StaticArray2D<float> aoSurface;
};
template <typename TDAG>
__global__ void trace_ambient_occlusion(TraceAmbientOcclusionParams params, const TDAG dag);

struct TraceLightingParams {
    // In
    double3 cameraPosition; // Reconstruct camera ray for fog computation.
    double3 rayMin;
    double3 rayDDx;
    double3 rayDDy;

    ToolInfo toolInfo;

    float sunBrightness;
    bool applyAmbientOcclusion;
    bool applyShadows;
    float fogDensity;
    StaticArray2D<float> aoSurface;
    StaticArray2D<float> sunLightSurface;
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;

    // Out
    StaticArray2D<uint32_t> finalColorsSurface;
};
__global__ void trace_lighting(TraceLightingParams params);

struct TracePathTracingParams {
    // In
    double3 cameraPosition; // Reconstruct camera ray for fog computation.
    double3 rayMin;
    double3 rayDDx;
    double3 rayDDy;
    bool integratePixel;
    float environmentBrightness;

    uint64_t randomSeed;
    uint32_t numSamples;
    uint32_t maxPathDepth;
    ColorParams colorParams;
    ToolInfo toolInfo;
    StaticArray2D<SurfaceInteraction> surfaceInteractionSurface;

    // In/out
    uint32_t numAccumulatedSamples;
    StaticArray2D<float3> accumulationBuffer;

    // Out
    StaticArray2D<uint32_t> finalColorsSurface;
};
template <typename TDAG, typename TDAGColors>
__global__ void trace_path_tracing(TracePathTracingParams params, const TDAG dag, const TDAGColors colors);

}