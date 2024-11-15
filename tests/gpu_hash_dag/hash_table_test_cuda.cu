#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/atomic64_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/compact_acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/ticket_board_hash_table.h"
#include "hash_table_test_cuda.h"

template <typename T>
__global__ void markActiveElements_kernel(T hashTable, gsl::span<const uint32_t> locations)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= locations.size())
        return;

    hashTable.markAsActive(locations[globalThreadIdx]);
}

template <typename T>
void markActiveElements_cuda(T& hashTable, gsl::span<const uint32_t> locations)
{
    markActiveElements_kernel<<<1024, 32>>>(hashTable, locations);
}

template <typename T>
static __global__ void add_kernel(T table, gsl::span<const uint32_t> items, gsl::span<uint32_t> outLocations)
{
    for (uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; globalThreadIdx < outLocations.size(); globalThreadIdx += gridDim.x * blockDim.x) {
        check(globalThreadIdx * table.itemSizeInU32 < items.size());
        const uint32_t* pNeedle = &items[globalThreadIdx * table.itemSizeInU32];
        outLocations[globalThreadIdx] = table.add(pNeedle);
    }
}
template <typename T>
static __global__ void addAsWarp_kernel(T table, gsl::span<const uint32_t> items, gsl::span<uint32_t> outLocations)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < outLocations.size() + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx < outLocations.size() && warpIdx * table.itemSizeInU32 < items.size()) {
            const uint32_t* pNeedle = &items[warpIdx * table.itemSizeInU32];
            outLocations[warpIdx] = table.addAsWarp(pNeedle);
        }
    }
}

template <typename T>
void add_cuda(T& hashTable, gsl::span<const uint32_t> elements, gsl::span<uint32_t> outLocations)
{
    add_kernel<<<128, 32>>>(hashTable, elements, outLocations);
    CUDA_CHECK_ERROR();
    cudaDeviceSynchronize();
}
template <typename T>
void addAsWarp_cuda(T& hashTable, gsl::span<const uint32_t> elements, gsl::span<uint32_t> outLocations)
{
    addAsWarp_kernel<<<128, 32>>>(hashTable, elements, outLocations);
    CUDA_CHECK_ERROR();
    cudaDeviceSynchronize();
}

template <typename T>
static __global__ void find_kernel(T table, gsl::span<const uint32_t> items, gsl::span<uint32_t> outLocations)
{
    for (uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; globalThreadIdx < outLocations.size(); globalThreadIdx += gridDim.x * blockDim.x) {
        check(globalThreadIdx * table.itemSizeInU32 < items.size());
        const uint32_t* pNeedle = &items[globalThreadIdx * table.itemSizeInU32];
        outLocations[globalThreadIdx] = table.find(pNeedle);
    }
}
template <typename T>
static __global__ void findAsWarp_kernel(T table, gsl::span<const uint32_t> items, gsl::span<uint32_t> outLocations)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < outLocations.size() + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx < outLocations.size() && warpIdx * table.itemSizeInU32 < items.size()) {
            const uint32_t* pNeedle = &items[warpIdx * table.itemSizeInU32];
            outLocations[warpIdx] = table.findAsWarp(pNeedle);
        }
    }
}

template <typename T>
void find_cuda(T& hashTable, gsl::span<const uint32_t> elements, gsl::span<uint32_t> outLocations)
{
    find_kernel<<<128, 32>>>(hashTable, elements, outLocations);
    CUDA_CHECK_ERROR();
    cudaDeviceSynchronize();
}
template <typename T>
void findAsWarp_cuda(T& hashTable, gsl::span<const uint32_t> elements, gsl::span<uint32_t> outLocations)
{
    findAsWarp_kernel<<<128, 32>>>(hashTable, elements, outLocations);
    CUDA_CHECK_ERROR();
    cudaDeviceSynchronize();
}

#define EXPLICIT_INSTANTIATE_TABLE_FUNCTIONS(Table)                                                                 \
    template void add_cuda(Table<EMemoryType::GPU_Malloc>&, gsl::span<const uint32_t>, gsl::span<uint32_t>);        \
    template void addAsWarp_cuda(Table<EMemoryType::GPU_Malloc>&, gsl::span<const uint32_t>, gsl::span<uint32_t>);  \
    template void find_cuda(Table<EMemoryType::GPU_Malloc>&, gsl::span<const uint32_t>, gsl::span<uint32_t>);       \
    template void findAsWarp_cuda(Table<EMemoryType::GPU_Malloc>&, gsl::span<const uint32_t>, gsl::span<uint32_t>); \
    template void markActiveElements_cuda(Table<EMemoryType::GPU_Malloc>&, gsl::span<const uint32_t>);

EXPLICIT_INSTANTIATE_TABLE_FUNCTIONS(Atomic64HashTable)
EXPLICIT_INSTANTIATE_TABLE_FUNCTIONS(AccelerationHashTable)
EXPLICIT_INSTANTIATE_TABLE_FUNCTIONS(CompactAccelerationHashTable)
EXPLICIT_INSTANTIATE_TABLE_FUNCTIONS(TicketBoardHashTable)
