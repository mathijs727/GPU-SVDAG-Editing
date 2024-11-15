//#include "cuda_helpers_cpp.h"
#include "cuda_helpers.h"
#include <cuda.h>

static __global__ void deviceMemset32Async_kernel(gsl::span<uint32_t> data, uint32_t value)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < data.size())
        data[globalThreadIdx] = value;
}

void deviceMemset32Async(uint32_t* dataU32, uint32_t value, size_t sizeInU32, cudaStream_t stream /*= 0*/)
{
    deviceMemset32Async_kernel<<<computeNumWorkGroups(sizeInU32), workGroupSize, 0, stream>>>(gsl::make_span(dataU32, sizeInU32), value);
    // cudaMemsetAsync(dataU32, value, sizeInU32, stream);
}

void cudaDeviceSynchronizeOnWindows()
{
#ifdef WIN32
    cudaDeviceSynchronize();
#endif
}
