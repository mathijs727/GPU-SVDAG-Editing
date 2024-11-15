#include <cuda.h>
#pragma warning(disable : 4324)
#include <cub/device/device_scan.cuh>
#include <thrust/iterator/transform_output_iterator.h>
#pragma warning(default : 4324)
#include "cub_scan.h"

template <typename T>
struct minus_one {
    __host__ __device__ T operator()(T v) { return v - 1; }
};

template <typename T>
void cubInclusiveSumMinusOne(gsl::span<const T> items, gsl::span<T> result, cudaStream_t stream)
{
    size_t tmpStorageSize = 0;
    void* pTmpStorage = nullptr;
    cub::DeviceScan::InclusiveSum(pTmpStorage, tmpStorageSize, thrust::raw_pointer_cast(items.data()), thrust::make_transform_output_iterator(std::begin(result), minus_one<uint32_t>()), (int)items.size(), stream);
    cudaMallocAsync(&pTmpStorage, tmpStorageSize, stream);
    cub::DeviceScan::InclusiveSum(pTmpStorage, tmpStorageSize, thrust::raw_pointer_cast(items.data()), thrust::make_transform_output_iterator(std::begin(result), minus_one<uint32_t>()), (int)items.size(), stream);
    cudaFreeAsync(pTmpStorage, stream);
}

template <typename T>
void cubExclusiveSum(gsl::span<const T> items, gsl::span<T> result, cudaStream_t stream)
{
    size_t tmpStorageSize = 0;
    void* pTmpStorage = nullptr;
    cub::DeviceScan::ExclusiveSum(pTmpStorage, tmpStorageSize, thrust::raw_pointer_cast(items.data()), thrust::raw_pointer_cast(result.data()), (int)items.size(), stream);
    cudaMallocAsync(&pTmpStorage, tmpStorageSize, stream);
    cub::DeviceScan::ExclusiveSum(pTmpStorage, tmpStorageSize, thrust::raw_pointer_cast(items.data()), thrust::raw_pointer_cast(result.data()), (int)items.size(), stream);
    cudaFreeAsync(pTmpStorage, stream);
}

template void cubInclusiveSumMinusOne(gsl::span<const uint32_t>, gsl::span<uint32_t>, cudaStream_t);
template void cubExclusiveSum(gsl::span<const uint32_t>, gsl::span<uint32_t>, cudaStream_t);
