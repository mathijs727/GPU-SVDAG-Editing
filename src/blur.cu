#include "blur.h"
#include "cuda_error_check.h"
#include "cuda_math.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

// https://en.wikipedia.org/wiki/Gaussian_function
static HOST_DEVICE float gaussian(float mean, float variance, float x)
{
    const float normalizationFactor = 1.0f / std::sqrt(2 * std::numbers::pi_v<float> * variance);
    return normalizationFactor * std::exp(-0.5f * (x - mean) * (x - mean) / variance);
}

BlurKernel::BlurKernel(uint32_t halfKernelSize)
    : halfKernelSize(halfKernelSize)
{
    // Match OpenCV:
    // https://theailearner.com/tag/cv2-getgaussiankernel/
    const float sigma = 0.3f * (halfKernelSize - 1) + 0.8f;
    std::vector<float> kernel((size_t)(2 * halfKernelSize + 1));
    for (size_t x = 0; x < kernel.size(); ++x) {
        kernel[x] = gaussian(halfKernelSize + 0.5f, sigma, (float)x);
    }
    // Normalize kernel such that it sums up to 1.0
    const float sum = std::reduce(std::begin(kernel), std::end(kernel), 0.0f, std::plus<float>());
    std::transform(std::begin(kernel), std::end(kernel), std::begin(kernel), [&](float v) { return v / sum; });

    kernelWeights = StaticArray<float>::allocate("Gaussian Kernel", kernel, EMemoryType::GPU_Malloc);
}

BlurKernel::~BlurKernel()
{
    kernelWeights.free();
}

static __global__ void horizontalKernel(StaticArray2D<float> input, StaticArray2D<float> output, StaticArray<float> kernelWeights, int halfKernelSize)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= input.width || pixel.y >= input.height)
        return;

    float sum = 0.0f;
    for (int dx = -halfKernelSize; dx <= halfKernelSize; ++dx) {
        const float weight = kernelWeights[dx + halfKernelSize];
        const int nx = pixel.x + dx;
        if (nx >= 0 && nx < input.width)
            sum += weight * input.read((uint32_t)nx, pixel.y);
    }
    output.write(pixel.x, pixel.y, sum);
}

static __global__ void verticalKernel(StaticArray2D<float> input, StaticArray2D<float> output, StaticArray<float> kernelWeights, int halfKernelSize)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= input.width || pixel.y >= input.height)
        return;

    float sum = 0.0f;
    for (int dy = -halfKernelSize; dy <= halfKernelSize; ++dy) {
        const float weight = kernelWeights[dy + halfKernelSize];
        const int ny = pixel.y + dy;
        if (ny >= 0 && ny < input.height)
            sum += weight * input.read(pixel.x, (uint32_t)ny);
    }
    output.write(pixel.x, pixel.y, sum);
}

void BlurKernel::apply(StaticArray2D<float> values, StaticArray2D<float> scratch) const
{
    constexpr uint32_t workGroupSize = 8;
    checkAlways(values.width % workGroupSize == 0);
    checkAlways(values.height % workGroupSize == 0);
    checkAlways(values.width == scratch.width);
    checkAlways(values.height == scratch.height);

    const uint32_t gridWidth = values.width / workGroupSize;
    const uint32_t gridHeight = values.height / workGroupSize;
    const dim3 gridSize { gridWidth, gridHeight, 1 };
    const dim3 workGroupSize2 { workGroupSize, workGroupSize, 1 };
    horizontalKernel<<<gridSize, workGroupSize2>>>(values, scratch, kernelWeights, halfKernelSize);
    CUDA_CHECK_ERROR();
    verticalKernel<<<gridSize, workGroupSize2>>>(scratch, values, kernelWeights, halfKernelSize);
    CUDA_CHECK_ERROR();
}

static __global__ void edgeAwareHorizontalKernel(StaticArray2D<float> input, StaticArray2D<float> output, StaticArray2D<uint3> paths, StaticArray<float> kernelWeights, int halfKernelSize)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= input.width || pixel.y >= input.height)
        return;

    constexpr int distanceThreshold = 3;
    constexpr int distanceThreshold2 = distanceThreshold * distanceThreshold;

    const int3 path = make_int3(paths.read(pixel.x, pixel.y));
    float sum = 0.0f, weightSum = 0.0f;
    for (int dx = -halfKernelSize; dx <= halfKernelSize; ++dx) {
        float weight = kernelWeights[dx + halfKernelSize];
        const int nx = pixel.x + dx;
        if (nx >= 0 && nx < input.width) {
            const int3 neighbourPath = make_int3(paths.read((uint32_t)nx, pixel.y));
            const int3 pathDiff = path - neighbourPath;
            const int distance2 = dot(pathDiff, pathDiff);
            if (distance2 > distanceThreshold2)
                weight = 0.0f;

            sum += weight * input.read((uint32_t)nx, pixel.y);
            weightSum += weight;
        }
    }
    output.write(pixel.x, pixel.y, sum / weightSum);
}

static __global__ void edgeAwareVerticalKernel(StaticArray2D<float> input, StaticArray2D<float> output, StaticArray2D<uint3> paths, StaticArray<float> kernelWeights, int halfKernelSize)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= input.width || pixel.y >= input.height)
        return;

    constexpr int distanceThreshold = 3;
    constexpr int distanceThreshold2 = distanceThreshold * distanceThreshold;

    const int3 path = make_int3(paths.read(pixel.x, pixel.y));
    float sum = 0.0f, weightSum = 0.0f;
    for (int dy = -halfKernelSize; dy <= halfKernelSize; ++dy) {
        float weight = kernelWeights[dy + halfKernelSize];
        const int ny = pixel.y + dy;
        if (ny >= 0 && ny < input.height) {
            const int3 neighbourPath = make_int3(paths.read(pixel.x, (uint32_t)ny));
            const int3 pathDiff = path - neighbourPath;
            const int distance2 = dot(pathDiff, pathDiff);
            if (distance2 > distanceThreshold2)
                weight = 0.0f;

            sum += weight * input.read(pixel.x, (uint32_t)ny);
            weightSum += weight;
        }
    }
    output.write(pixel.x, pixel.y, sum / weightSum);
}

void BlurKernel::applyEdgeAware(StaticArray2D<float> values, StaticArray2D<float> scratch, StaticArray2D<uint3> paths) const
{
    constexpr uint32_t workGroupSize = 8;
    checkAlways(values.width % workGroupSize == 0);
    checkAlways(values.height % workGroupSize == 0);
    checkAlways(values.width == scratch.width);
    checkAlways(values.height == scratch.height);

    const uint32_t gridWidth = values.width / workGroupSize;
    const uint32_t gridHeight = values.height / workGroupSize;
    const dim3 gridSize { gridWidth, gridHeight, 1 };
    const dim3 workGroupSize2 { workGroupSize, workGroupSize, 1 };
    edgeAwareHorizontalKernel<<<gridSize, workGroupSize2>>>(values, scratch, paths, kernelWeights, halfKernelSize);
    CUDA_CHECK_ERROR();
    edgeAwareVerticalKernel<<<gridSize, workGroupSize2>>>(scratch, values, paths, kernelWeights, halfKernelSize);
    CUDA_CHECK_ERROR();
}
