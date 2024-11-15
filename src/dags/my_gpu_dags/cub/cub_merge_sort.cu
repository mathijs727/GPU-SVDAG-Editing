#include <cuda.h>
#pragma warning(disable : 4324)
// #include <cub/device/device_scan.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/iterator/transform_output_iterator.h>
#pragma warning(default : 4324)
//
#include "cub_merge_sort.h"
#include "cuda_error_check.h"
#include "cuda_helpers_cpp.h"

#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/individual_chaining_hash_table.h"

template <typename T1, typename T2>
void cubDeviceMergeSortPairs(gsl::span<T1> keys, gsl::span<T2> items, cudaStream_t stream)
{
    size_t requiredMemorySize = 0;
    cub::DeviceMergeSort::SortPairs(nullptr, requiredMemorySize, thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(items.data()), keys.size(), thrust::less<T1>(), stream);
    void* pMemory = nullptr;
    cudaMallocAsync(&pMemory, requiredMemorySize, stream);
    cub::DeviceMergeSort::SortPairs(pMemory, requiredMemorySize, thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(items.data()), keys.size(), thrust::less<T1>(), stream);
    cudaFreeAsync(pMemory, stream);
}

//template void cubDeviceMergeSortPairs(gsl::span<typename IntermediateSVO::Node>, gsl::span<uint32_t>, cudaStream_t);
//template void cubDeviceMergeSortPairs(gsl::span<typename IntermediateSVO::Leaf>, gsl::span<uint32_t>, cudaStream_t);

/* template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<2>>, gsl::span<uint32_t>, cudaStream_t);
template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<3>>, gsl::span<uint32_t>, cudaStream_t);
template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<4>>, gsl::span<uint32_t>, cudaStream_t);
template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<5>>, gsl::span<uint32_t>, cudaStream_t);
template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<6>>, gsl::span<uint32_t>, cudaStream_t);
template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<7>>, gsl::span<uint32_t>, cudaStream_t);
template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<8>>, gsl::span<uint32_t>, cudaStream_t);
template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<9>>, gsl::span<uint32_t>, cudaStream_t);
template void cubDeviceMergeSortPairs(gsl::span<MyGpuDagElement<10>>, gsl::span<uint32_t>, cudaStream_t);*/
