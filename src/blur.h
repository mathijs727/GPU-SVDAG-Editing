#pragma once
#include "array.h"
#include "array2d.h"

struct BlurKernel {
    BlurKernel(uint32_t halfKernelSize);
    ~BlurKernel();

    void apply(StaticArray2D<float> values, StaticArray2D<float> scratch) const;
    void applyEdgeAware(StaticArray2D<float> values, StaticArray2D<float> scratch, StaticArray2D<uint3> paths) const;

private:
    uint32_t halfKernelSize;
    StaticArray<float> kernelWeights;
};

