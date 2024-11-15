#include "typedefs.h"
//
#include "cuda_error_check.h"
#include "cuda_helpers.h"
#include "dag_tracer.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/dag_utils.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "events.h"
#include "image.h"
#include "memory.h"
#include "tracer.h"

DAGTracer::DAGTracer(bool headLess, EventsManager* eventsManager)
    : headLess(headLess)
    , eventsManager(eventsManager)
    , ambientOcclusionBlurKernel(4)
{
    colorsBuffer = StaticArray2D<uint32_t>::allocate("colors buffer", imageWidth, imageHeight);
    surfaceInteractionBuffer = StaticArray2D<Tracer::SurfaceInteraction>::allocate("SurfaceInteraction buffer", imageWidth, imageHeight);
    ambientOcclusionBuffer = StaticArray2D<float>::allocate("ambient occlusion buffer", imageWidth, imageHeight);
    ambientOcclusionBlurScratchBuffer_sunLightBuffer = StaticArray2D<float>::allocate("ambient occlusion buffer", imageWidth, imageHeight);
    pathTracingAccumulationBuffer = StaticArray2D<float3>::allocate("path tracing accumulation buffer", imageWidth, imageHeight);

    if (!headLess) {
        glGenTextures(1, &colorsImage);
        glBindTexture(GL_TEXTURE_2D, colorsImage);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int32)imageWidth, (int32)imageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_2D, 0);
        colorsSurface = GLSurface2D::create(colorsImage);
    }
    // We cannot use GPU managed memory to read data from the GPU without requiring a full device synchronization on Windows:
    // "Applications running on Windows (whether in TCC or WDDM mode) will use the basic Unified Memory model as on
    //  pre-6.x architectures even when they are running on hardware with compute capability 6.x or higher."
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
    pathCache = Memory::malloc<ToolPath>("path cache", pathCacheSize * sizeof(ToolPath), EMemoryType::CPU);
}

DAGTracer::~DAGTracer()
{
    if (!headLess) {
        colorsSurface.free();
        glDeleteTextures(1, &colorsImage);

        Memory::free(pathCache);
    }

    surfaceInteractionBuffer.free();
    colorsBuffer.free();
    ambientOcclusionBuffer.free();
    ambientOcclusionBlurScratchBuffer_sunLightBuffer.free();
    pathTracingAccumulationBuffer.free();
}

inline Tracer::TracePathsParams get_trace_params(
    const CameraView& camera,
    uint32 levels,
    const DAGInfo& dagInfo)
{
    const double3 position = make_double3(camera.position);
    const double3 direction = make_double3(camera.forward());
    const double3 up = make_double3(camera.up());
    const double3 right = make_double3(camera.right());

    const double3 boundsMin = make_double3(dagInfo.boundsAABBMin);
    const double3 boundsMax = make_double3(dagInfo.boundsAABBMax);

    const double fov = camera.fov / 2.0 * (double(M_PI) / 180.);
    const double aspect_ratio = double(imageWidth) / double(imageHeight);

    const double3 X = right * sin(fov) * aspect_ratio;
    const double3 Y = up * sin(fov);
    const double3 Z = direction * cos(fov);

    const double3 bottomLeft = position + Z - Y - X;
    const double3 bottomRight = position + Z - Y + X;
    const double3 topLeft = position + Z + Y - X;

    const double3 translation = -boundsMin;
    const double3 scale = make_double3(double(1 << levels)) / (boundsMax - boundsMin);

    const double3 finalPosition = (position + translation) * scale;
    const double3 finalBottomLeft = (bottomLeft + translation) * scale;
    const double3 finalTopLeft = (topLeft + translation) * scale;
    const double3 finalBottomRight = (bottomRight + translation) * scale;
    const double3 dx = (finalBottomRight - finalBottomLeft) * (1.0 / imageWidth);
    const double3 dy = (finalTopLeft - finalBottomLeft) * (1.0 / imageHeight);

    Tracer::TracePathsParams params;

    params.cameraPosition = finalPosition;
    params.rayMin = finalBottomLeft;
    params.rayDDx = dx;
    params.rayDDy = dy;

    return params;
}

inline Tracer::ColorParams get_color_params(const VoxelTextures& voxelTextures, EDebugColors debugColors, uint32 debugColorsIndexLevel)
{
    return Tracer::ColorParams {
        .materialTextures = voxelTextures.gpuMaterials,
        .debugColors = debugColors,
        .debugColorsIndexLevel = debugColorsIndexLevel
    };
}

template <typename TDAG>
std::chrono::duration<double> DAGTracer::resolve_paths(const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag)
{
    PROFILE_FUNCTION();

    const dim3 block_dim = dim3(4, 64);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    auto traceParams = get_trace_params(camera, dag.levels, dagInfo);
    traceParams.surfaceInteractionSurface = surfaceInteractionBuffer;

    CUDA_CHECK_ERROR();

    {
        auto t = eventsManager->createTiming("resolve_paths");
        Tracer::trace_paths<<<grid_dim, block_dim>>>(traceParams, dag);
        CUDA_CHECK_ERROR();
    }

    CUDA_CHECK_ERROR();

    return eventsManager->getLastCompletedTiming("resolve_paths");
}

template <typename TDAG, typename TDAGColors>
std::chrono::duration<double> DAGTracer::resolve_colors(const TDAG& dag, const TDAGColors& colors, const VoxelTextures& voxelTextures, EDebugColors debugColors, uint32 debugColorsIndexLevel)
{
    PROFILE_FUNCTION();

    colors.check_ready_for_rt();

    const dim3 block_dim = dim3(4, 64);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    Tracer::TraceColorsParams traceParams;
    traceParams.colorParams = get_color_params(voxelTextures, debugColors, debugColorsIndexLevel);
    traceParams.surfaceInteractionSurface = surfaceInteractionBuffer;

    CUDA_CHECK_ERROR();

    {
        auto t = eventsManager->createTiming("trace_colors");
        Tracer::trace_colors<<<grid_dim, block_dim>>>(traceParams, dag, colors);
        CUDA_CHECK_ERROR();
    }

    CUDA_CHECK_ERROR();

    return eventsManager->getLastCompletedTiming("trace_colors");
}

template <typename TDAG>
std::chrono::duration<double> DAGTracer::resolve_ambient_occlusion(const TDAG& dag, uint32_t numSamples, float shadowBias, float aoRayLength)
{
    PROFILE_FUNCTION();

    const dim3 block_dim = dim3(4, 64);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    Tracer::TraceAmbientOcclusionParams traceParams {
        .numSamples = numSamples,
        .shadowBias = shadowBias,
        .aoRayLength = aoRayLength,
        .randomSeed = ambientOcclusionRandomSeed++,
        .surfaceInteractionSurface = surfaceInteractionBuffer,
        .aoSurface = ambientOcclusionBuffer,
    };

    {
        auto t = eventsManager->createTiming("resolve_ambient_occlusion");
        Tracer::trace_ambient_occlusion<<<grid_dim, block_dim>>>(traceParams, dag);
        CUDA_CHECK_ERROR();
    }

    return eventsManager->getLastCompletedTiming("resolve_ambient_occlusion");
}

std::chrono::duration<double> DAGTracer::resolve_ambient_occlusion_blur()
{
    {
        auto t = eventsManager->createTiming("resolve_ambient_occlusion_blur");
        ambientOcclusionBlurKernel.apply(ambientOcclusionBuffer, ambientOcclusionBlurScratchBuffer_sunLightBuffer);
    }
    return eventsManager->getLastCompletedTiming("resolve_ambient_occlusion_blur");
}

template <typename TDAG>
std::chrono::duration<double> DAGTracer::resolve_shadows(const TDAG& dag, float shadowBias)
{
    PROFILE_FUNCTION();

    const dim3 block_dim = dim3(4, 64);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    Tracer::TraceShadowsParams traceParams {
        .shadowBias = shadowBias,
        .surfaceInteractionSurface = surfaceInteractionBuffer,
        .sunLightSurface = ambientOcclusionBlurScratchBuffer_sunLightBuffer
    };

    CUDA_CHECK_ERROR();

    {
        auto t = eventsManager->createTiming("resolve_shadows");
        Tracer::trace_shadows<<<grid_dim, block_dim>>>(traceParams, dag);
        CUDA_CHECK_ERROR();
    }

    return eventsManager->getLastCompletedTiming("resolve_shadows");
}

std::chrono::duration<double> DAGTracer::resolve_lighting(const CameraView& camera, const DAGInfo& dagInfo, const ToolInfo& toolInfo, bool applyAmbientOcclusion, bool applyShadows, float fogDensity)
{
    PROFILE_FUNCTION();

    const dim3 block_dim = dim3(4, 64);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    const auto cameraParams = get_trace_params(camera, MAX_LEVELS, dagInfo);

    Tracer::TraceLightingParams traceParams {
        .cameraPosition = cameraParams.cameraPosition,
        .rayMin = cameraParams.rayMin,
        .rayDDx = cameraParams.rayDDx,
        .rayDDy = cameraParams.rayDDy,
        .toolInfo = toolInfo,
        .applyAmbientOcclusion = applyAmbientOcclusion,
        .applyShadows = applyShadows,
        .fogDensity = fogDensity,
        .aoSurface = ambientOcclusionBuffer,
        .sunLightSurface = ambientOcclusionBlurScratchBuffer_sunLightBuffer,
        .surfaceInteractionSurface = surfaceInteractionBuffer,
        .finalColorsSurface = colorsBuffer
    };

    {
        auto t = eventsManager->createTiming("trace_lighting");
        Tracer::trace_lighting<<<grid_dim, block_dim>>>(traceParams);
        CUDA_CHECK_ERROR();
    }

    return eventsManager->getLastCompletedTiming("trace_lighting");
}

template <typename TDAG, typename TDAGColors>
std::chrono::duration<double> DAGTracer::resolve_path_tracing(
    const CameraView& camera, const DAGInfo& dagInfo, const TDAG& dag, const TDAGColors& colors,
    const VoxelTextures& voxelTextures, EDebugColors debugColors, uint32 debugColorsIndexLevel, const ToolInfo& toolInfo,
    float environmentBrightness, uint32_t numSamples, uint32_t maxPathDepth, bool integratePixel)
{
    PROFILE_FUNCTION();

    const dim3 block_dim = dim3(4, 64);
    const dim3 grid_dim = dim3(imageWidth / block_dim.x + 1, imageHeight / block_dim.y + 1);

    const auto cameraParams = get_trace_params(camera, MAX_LEVELS, dagInfo);
    Tracer::TracePathTracingParams traceParams;
    std::memset(&traceParams, 0, sizeof(traceParams)); // Prevent undefined behaviour by memcmp padding bytes.
    traceParams.cameraPosition = cameraParams.cameraPosition;
    traceParams.rayMin = cameraParams.rayMin;
    traceParams.rayDDx = cameraParams.rayDDx;
    traceParams.rayDDy = cameraParams.rayDDy;
    traceParams.integratePixel = integratePixel;
    traceParams.environmentBrightness = environmentBrightness;
    traceParams.randomSeed = ambientOcclusionRandomSeed++;
    traceParams.numSamples = numSamples;
    traceParams.maxPathDepth = maxPathDepth;
    traceParams.colorParams = get_color_params(voxelTextures, debugColors, debugColorsIndexLevel);
    traceParams.toolInfo = toolInfo;
    traceParams.surfaceInteractionSurface = surfaceInteractionBuffer;
    traceParams.numAccumulatedSamples = numAccumulatedSamples;
    traceParams.accumulationBuffer = pathTracingAccumulationBuffer;
    traceParams.finalColorsSurface = colorsBuffer;

    // Check if any of the parameters changed, and if so, clear the accumulation buffer.
    previousPathTraceParams.numAccumulatedSamples = traceParams.numAccumulatedSamples; // ignore changes
    previousPathTraceParams.randomSeed = traceParams.randomSeed; // ignore changes
    previousPathTraceParams.toolInfo = traceParams.toolInfo; // ignore changes
    if (memcmp(&traceParams, &previousPathTraceParams, sizeof(traceParams)) != 0 || dag.get_first_node_index() != previousDagRootNode) {
        cudaMemsetAsync(pathTracingAccumulationBuffer.data(), 0, pathTracingAccumulationBuffer.size_in_bytes(), nullptr);
        traceParams.numAccumulatedSamples = numAccumulatedSamples = 0;
    }
    memset(&previousPathTraceParams, 0, sizeof(previousPathTraceParams));
    previousPathTraceParams = traceParams;
    previousDagRootNode = dag.get_first_node_index();

    // After 256 paths the image should be sharp enough.
    // Stop tracing to save compute resources and to prevent precision/overflow issues with the accumulation buffer.
    // We do run the kernel so we can update the tool overlay.
    if (numAccumulatedSamples > 256) {
        traceParams.numSamples = 0;
    }

    {
        auto t = eventsManager->createTiming("trace_path_tracing");
        Tracer::trace_path_tracing<<<grid_dim, block_dim>>>(traceParams, dag, colors);
        CUDA_CHECK_ERROR();
    }

    // Add the number of samples that we have rendered.
    numAccumulatedSamples += traceParams.numSamples;

    return eventsManager->getLastCompletedTiming("trace_path_tracing");
}

__global__ void read_path(uint2 pixel, StaticArray2D<Tracer::SurfaceInteraction> surfaceInteractions, ToolPath* pOutput)
{
    const auto& si = surfaceInteractions.read(pixel);
    const uint3 centerVoxel = si.path.path;
    const uint3 neighbourVoxel = make_uint3(make_int3(centerVoxel) + make_int3(si.normal));
    pOutput->centerPath = centerVoxel;
    pOutput->neighbourPath = neighbourVoxel;
}

ToolPath DAGTracer::get_path(const uint2& pixelPos)
{
    PROFILE_FUNCTION();
    cudaStream_t stream = nullptr;

    if (headLess)
        return {};

    check(pixelPos.x < imageWidth);
    check(pixelPos.y < imageHeight);

    // Enqueue the read_path kernel which reads the 3D position of the voxel at pixel x/y
    CUDA_CHECK_ERROR();
    ++currentPathIndex;

    // cudaMemcpyAsync(&pathCache[currentPathIndex % pathCacheSize], pathsBuffer.getPixelPointer(posX, posY), sizeof(uint3), cudaMemcpyDeviceToHost);
    ToolPath* pTmp;
    cudaMallocAsync(&pTmp, sizeof(ToolPath), stream);
    CUDA_CHECK_ERROR();

    read_path<<<1, 1, 0, stream>>>(pixelPos, surfaceInteractionBuffer, pTmp);
    CUDA_CHECK_ERROR();

    cudaMemcpyAsync(&pathCache[currentPathIndex % pathCacheSize], pTmp, sizeof(ToolPath), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(pTmp, stream);
    CUDA_CHECK_ERROR();

    eventsManager->insertFenceValue("get_path", currentPathIndex);
    CUDA_CHECK_ERROR();

    // The GPU may lag behind the CPU so take the position read for the last completed frame.
    const auto lastCompletedPathIndex = eventsManager->getLastCompletedFenceValue("get_path");
    return pathCache[lastCompletedPathIndex % pathCacheSize];
}

template <typename TDAG>
__global__ void getVoxelValues_kernel(const TDAG dag, gsl::span<const uint3> locations, gsl::span<uint32_t> outVoxelMaterials)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= locations.size())
        return;

    Path path;
    path.path = locations[globalThreadIdx];
    auto optValue = DAGUtils::get_value(dag, path);
    outVoxelMaterials[globalThreadIdx] = optValue.value_or((uint32_t)-1);
}

template <typename TDAG>
void DAGTracer::get_voxel_values(const TDAG& dag, gsl::span<const uint3> locations, gsl::span<uint32_t> outVoxelMaterials) const
{
    check(locations.size() == outVoxelMaterials.size());
    cudaStream_t stream = nullptr;
    auto locationsGPU = cudaMallocAsyncRange<uint3>(locations.size(), stream);
    cudaMemcpyAsync(locationsGPU.data(), locations.data(), locations.size_bytes(), cudaMemcpyHostToDevice, stream);
    auto voxelMaterialsGPU = cudaMallocAsyncRange<uint32_t>(outVoxelMaterials.size(), stream);

    getVoxelValues_kernel<TDAG><<<computeNumWorkGroups(locations.size()), workGroupSize, 0, stream>>>(dag, locationsGPU, voxelMaterialsGPU);
    CUDA_CHECK_ERROR();

    cudaMemcpyAsync(outVoxelMaterials.data(), voxelMaterialsGPU.data(), outVoxelMaterials.size_bytes(), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(locationsGPU.data(), stream);
    cudaFreeAsync(voxelMaterialsGPU.data(), stream);
    cudaStreamSynchronize(stream);
}

uint32_t DAGTracer::get_path_tracing_num_accumulated_samples() const
{
    return numAccumulatedSamples;
}

#define DAG_IMPLS(Dag)                                                                                                   \
    template void DAGTracer::get_voxel_values<Dag>(const Dag&, gsl::span<const uint3>, gsl::span<uint32_t>) const;       \
    template std::chrono::duration<double> DAGTracer::resolve_paths<Dag>(const CameraView&, const DAGInfo&, const Dag&); \
    template std::chrono::duration<double> DAGTracer::resolve_shadows<Dag>(const Dag&, float);                           \
    template std::chrono::duration<double> DAGTracer::resolve_ambient_occlusion<Dag>(const Dag&, uint32_t, float, float);

#define DAG_COLOR_IMPLS(Dag, Colors)                                                                                                                      \
    template std::chrono::duration<double> DAGTracer::resolve_colors<Dag, Colors>(const Dag&, const Colors&, const VoxelTextures&, EDebugColors, uint32); \
    template std::chrono::duration<double> DAGTracer::resolve_path_tracing<Dag, Colors>(const CameraView&, const DAGInfo&, const Dag&, const Colors&, const VoxelTextures&, EDebugColors, uint32, const ToolInfo&, float, uint32_t, uint32_t, bool);

using MyGPUHashDAG_T = MyGPUHashDAG<EMemoryType::GPU_Malloc>;

DAG_IMPLS(BasicDAG)
DAG_IMPLS(HashDAG)
DAG_IMPLS(MyGPUHashDAG_T)

DAG_COLOR_IMPLS(BasicDAG, BasicDAGUncompressedColors)
DAG_COLOR_IMPLS(BasicDAG, BasicDAGCompressedColors)
DAG_COLOR_IMPLS(BasicDAG, BasicDAGColorErrors)
DAG_COLOR_IMPLS(HashDAG, HashDAGColors)
DAG_COLOR_IMPLS(MyGPUHashDAG_T, HashDAGColors)
