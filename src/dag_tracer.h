#pragma once
#include "array2d.h"
#include "blur.h"
#include "camera_view.h"
#include "cuda_gl_buffer.h"
#include "dag_info.h"
#include "image.h"
#include "opengl_interop.h"
#include "tracer.h"
#include "typedefs.h"
#include <chrono>

class EventsManager;
struct VoxelTextures;

class DAGTracer {
public:
    const bool headLess;

    DAGTracer(bool headLess, EventsManager* eventsManager);
    ~DAGTracer();

    inline GLuint get_colors_image() const
    {
        return colorsImage;
    }
    inline void update_colors_image()
    {
        colorsSurface.copyFrom(colorsBuffer);
    }

    template <typename TDAG>
    std::chrono::duration<double> resolve_paths(const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag);
    template <typename TDAG, typename TDAGColors>
    std::chrono::duration<double> resolve_colors(const TDAG& dag, const TDAGColors& colors, const VoxelTextures& voxelTextures, EDebugColors debugColors, uint32 debugColorsIndexLevel);
    // WARNING: Ambient Occlusion **MUST** run before shadows because they share a buffer.
    template <typename TDAG>
    std::chrono::duration<double> resolve_ambient_occlusion(const TDAG& dag, uint32_t numSamples, float shadowBias, float aoRayLength);
    std::chrono::duration<double> resolve_ambient_occlusion_blur();
    template <typename TDAG>
    std::chrono::duration<double> resolve_shadows(const TDAG& dag, float shadowBias);
    std::chrono::duration<double> resolve_lighting(const CameraView& camera, const DAGInfo& dagInfo, const ToolInfo& toolInfo, bool applyAmbientOcclusion, bool applyShadows, float fogDensity);

    template <typename TDAG, typename TDAGColors>
    std::chrono::duration<double> resolve_path_tracing(
        const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag, const TDAGColors& colors,
        const VoxelTextures& voxelTextures, EDebugColors debugColors, uint32 debugColorsIndexLevel, const ToolInfo& toolInfo,
        float environmentBrightness, uint32_t numSamples, uint32_t maxPathDepth, bool integratePixel);
    ToolPath get_path(const uint2& pixelPos);

    template <typename TDAG>
    void get_voxel_values(const TDAG& dag, gsl::span<const uint3> locations, gsl::span<uint32_t> outVoxelMaterials) const;

    uint32_t get_path_tracing_num_accumulated_samples() const;

private:
    GLuint colorsImage = 0;

    StaticArray2D<Tracer::SurfaceInteraction> surfaceInteractionBuffer;
    StaticArray2D<uint32_t> colorsBuffer;
    GLSurface2D colorsSurface;

    StaticArray2D<float> ambientOcclusionBuffer;
    StaticArray2D<float> ambientOcclusionBlurScratchBuffer_sunLightBuffer;
    BlurKernel ambientOcclusionBlurKernel;
    uint32_t ambientOcclusionRandomSeed = 123;

    StaticArray2D<float3> pathTracingAccumulationBuffer;
    Tracer::TracePathTracingParams previousPathTraceParams;
    uint32_t previousDagRootNode = (uint32_t)-1;
    uint32_t numAccumulatedSamples = 0;

    static constexpr size_t pathCacheSize = 16;
    size_t currentPathIndex = 0;
    ToolPath* pathCache;
    EventsManager* eventsManager;
};
