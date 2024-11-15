#include <cuda.h>
#pragma warning(disable : 4324)
#include <cub/device/device_select.cuh>
#pragma warning(default : 4324)
#include "cub_select.h"

template <typename T>
void cubDeviceSelectUnique(gsl::span<const T> inKeys, gsl::span<T> outKeys, uint32_t* pNumOutKeys, cudaStream_t stream)
{
    size_t tmpStorageSize = 0;
    void* pTmpStorage = nullptr;
    cub::DeviceSelect::Unique(pTmpStorage, tmpStorageSize, inKeys.data(), outKeys.data(), pNumOutKeys, outKeys.size(), stream);
    cudaMallocAsync(&pTmpStorage, tmpStorageSize, stream);
    cub::DeviceSelect::Unique(pTmpStorage, tmpStorageSize, inKeys.data(), outKeys.data(), pNumOutKeys, outKeys.size(), stream);
    cudaFreeAsync(pTmpStorage, stream);
}

template void cubDeviceSelectUnique(gsl::span<const uint32_t>, gsl::span<uint32_t>, uint32_t*, cudaStream_t);
