#include "cuda_error_check.h"
#include "cuda_helpers.h"
#include "my_gpu_hash_dag_edits.h"
#include <cuda.h>
// Then CUDA
#include "dags/my_gpu_dags/create_edit_svo.h"
#include "dags/my_gpu_dags/cub/cub_merge_sort.h"
#include "dags/my_gpu_dags/cub/cub_scan.h"
#include "dags/my_gpu_dags/my_gpu_dag_editors.h"
#include "dags/my_gpu_dags/my_gpu_dag_item.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/duplicate_detection_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_edits.h"
#include "stats.h"
#include "timings.h"
#include "typedefs.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

static constexpr uint32_t maxWorkGroups = 8192 * 4;
static constexpr uint32_t warpsPerWorkGroup = 2;

static constexpr uint32_t not_found_sentinel = 0xFFFFFFFF;
static constexpr uint32_t empty_sentinel = 0xFFFFFFFE;

#define TIMING_SYNCHRONIZATION 0

static void deviceSynchronizeTiming()
{
#if TIMING_SYNCHRONIZATION
    cudaDeviceSynchronize();
#endif
}

struct NodeInformation {
    uint32_t handle;
    uint32_t fullyFilledMaterial;
};

static __global__ void updateChildPointers(
    const MyGPUHashDAG<EMemoryType::GPU_Malloc> inDag, uint32_t level,
    gsl::span<typename IntermediateSVO::Node> inOutNodes,
    gsl::span<const NodeInformation> inPrevLevelMapping,
    gsl::span<NodeInformation> outNextLevelMapping)
{
    const auto globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= inOutNodes.size())
        return;

    auto& node = inOutNodes[globalThreadIdx];
    const uint32_t nodeHeader = node.padding[0];
    uint32_t childMask = inDag.get_node_child_mask(nodeHeader);
    const uint32_t childIntermediateMask = inDag.get_node_user_payload(nodeHeader);

    uint32_t* pChildren = &node.padding[inDag.get_header_size()];
    uint32_t inChildOffset = 0, outChildOffset = 0;
    for (uint32_t childIdx = 0, childBit = 1; childIdx < 8; ++childIdx, childBit <<= 1) {
        if (childMask & childBit) {
            uint32_t childHandle = pChildren[inChildOffset++];
            if (childIntermediateMask & childBit) { // Is pointer to the IntermediateSVO (not the SVDAG)
                check(childHandle != sentinel);
                childHandle = inPrevLevelMapping[childHandle].handle;
            }

            if (childHandle != sentinel) {
                pChildren[outChildOffset++] = childHandle;
            } else {
                // If the child node was fully empty then we remove the child by flipping its bit from 1 to 0.
                childMask ^= childBit;
            }
        }
    }

    if (outChildOffset != inChildOffset) {
        // If one or more children where removed then we need to:
        //  * Re-encode the node header
        //  * Set the now unused bits to 0
        inDag.encode_node_header(&node.padding[0], childMask);

        const uint32_t newNodeSize = inDag.get_header_size() + outChildOffset;
        check(node.size == inDag.get_header_size() + inChildOffset);
        check(newNodeSize < node.size);
        for (uint32_t i = newNodeSize; i < node.size; ++i)
            node.padding[i] = 0;
        node.size = newNodeSize;
    } else {
        // If no children were removed then we still update the node header to clear the user bits.
        // This indicates that all pointers are now pointing to the DAG and not the SVO.
        inDag.encode_node_header(&node.padding[0], childMask);
    }
}
template <uint32_t level, typename T>
static __global__ void discover_fully_filled(
    gsl::span<T> inOutItems,
    gsl::span<const NodeInformation> inPrevLevelMapping,
    MyGPUHashDAG<EMemoryType::GPU_Malloc> inDag,
    gsl::span<NodeInformation> outCurLevelMapping,
    uint32_t* outHasFullyFilledNodes)
{
    const auto globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= inOutItems.size())
        return;

    uint32_t fullyFilledMaterial = sentinel;
    bool allSameMaterial = false;
    if constexpr (level == inDag.leaf_level()) {
        const auto& inLeaf = inOutItems[globalThreadIdx];
        allSameMaterial = inDag.is_leaf_fully_filled(inLeaf.padding.data(), fullyFilledMaterial);
    } else {
        auto& node = inOutItems[globalThreadIdx];
        const uint32_t nodeHeader = node.padding[0];
        if (inDag.get_node_child_mask(nodeHeader) == 0xFF) {
            const uint32_t childIntermediateMask = inDag.get_node_user_payload(nodeHeader);

            allSameMaterial = true;
            for (uint32_t i = 0; i < 8; ++i) {
                const bool isChildInIntermediateSvo = (childIntermediateMask >> i) & 1;

                const uint32_t childHandle = node.padding[inDag.get_header_size() + i];
                uint32_t childFullyFilledMaterial = sentinel;
                if (isChildInIntermediateSvo) {
                    childFullyFilledMaterial = inPrevLevelMapping[childHandle].fullyFilledMaterial;
                } else if (level + 1 == inDag.leaf_level()) {
                    allSameMaterial &= inDag.is_leaf_fully_filled(inDag.get_leaf_ptr(childHandle), childFullyFilledMaterial);
                } else {
                    allSameMaterial &= inDag.get_node_is_fully_filled(inDag.get_node_ptr(level, childHandle)[0], childFullyFilledMaterial);
                }

                if (i == 0)
                    fullyFilledMaterial = childFullyFilledMaterial;
                allSameMaterial &= (childFullyFilledMaterial == fullyFilledMaterial);
                if (!allSameMaterial)
                    break;
            } // for child 0..8

            if (allSameMaterial)
                inDag.encode_node_header(&node.padding[0], 0xFF, fullyFilledMaterial);
        } // child_mask == 0xFF
    }

    outCurLevelMapping[globalThreadIdx].fullyFilledMaterial = fullyFilledMaterial;
    if (allSameMaterial) {
        outCurLevelMapping[globalThreadIdx].handle = inDag.fullyFilledNodes.read(level, fullyFilledMaterial);
        atomicAdd(outHasFullyFilledNodes, 1);
    }
}

// clang-format off
template <typename T>
concept dag_has_warp_add = requires(T& dag, const uint32_t* pItem) {
    { dag.add_node_as_warp(pItem) };
    { dag.add_leaf_as_warp(pItem) };
};
template <typename T>
concept dag_has_warp_find = requires(T& dag, const uint32_t* pItem) {
    { dag.find_leaf_as_warp(pItem) };
};
// clang-format on

template <typename T, bool isLeafLevel, typename HashTable>
__global__ void insertDuplicatesIntoHashTableAsWarp_kernel(HashTable hashTable, gsl::span<const T> inItems, gsl::span<uint32_t> outIndices, uint32_t* pInItemsBytes)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();

    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < inItems.size() + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx < inItems.size()) {
            const auto& item = inItems[warpIdx];
            const uint32_t* pItem = &item.padding[0];

            bool isEmpty = item.size == 0;
            if constexpr (isLeafLevel)
                isEmpty |= pItem[0] == 0 && pItem[1] == 0;
            else
                isEmpty |= Utils::child_mask(pItem[0]) == 0;
            if (isEmpty) {
                outIndices[warpIdx] = empty_sentinel;
                continue;
            }

            hashTable.addAsWarp(pItem, warpIdx, outIndices[warpIdx]);
#if CAPTURE_MEMORY_STATS_SLOW
            if (warp.thread_rank() == 0)
                atomicAdd(pInItemsBytes, inItems[warpIdx].size * sizeof(uint32_t));
#endif
        }
    }
}

template <typename T, uint32_t level, typename HashTable>
__global__ void findUniqueInHashTableAsWarp_kernel1(
    const HashTable hashTable, const MyGPUHashDAG<EMemoryType::GPU_Malloc> hashDag,
    gsl::span<const T> items,
    gsl::span<uint32_t> indicesToUnique)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();

    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < items.size() + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx >= items.size())
            return;
        // Already marked as duplicate by insertion.
        if (indicesToUnique[warpIdx] != not_found_sentinel)
            continue;

        const uint32_t* pItem = &items[warpIdx].padding[0];
        if (auto idx = hashTable.findAsWarp(pItem); idx != warpIdx)
            indicesToUnique[warpIdx] = idx;
    }
}

template <typename T, uint32_t level, typename HashTable>
__global__ void findUniqueInHashTableAsWarp_kernel2(
    const HashTable hashTable, const MyGPUHashDAG<EMemoryType::GPU_Malloc> hashDag,
    gsl::span<const T> items,
    gsl::span<const uint32_t> indicesToUnique, gsl::span<uint32_t> outDagHandles)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    const auto threadRank = warp.thread_rank();

    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < items.size() + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx >= items.size())
            return;
        // Marked as duplicate in the hash table; no point in searching for it.
        if (indicesToUnique[warpIdx] != not_found_sentinel)
            continue;

        const uint32_t* pItem = &items[warpIdx].padding[0];
        if constexpr (level == std::remove_all_extents_t<decltype(hashDag)>::leaf_level()) {
            outDagHandles[warpIdx] = hashDag.find_leaf_as_warp(pItem);
        } else {
            outDagHandles[warpIdx] = hashDag.find_node_as_warp(pItem);
        }
    }
}

template <typename T, uint32_t level>
__global__ void insertUniqueInHashTableAsWarp_kernel(
    MyGPUHashDAG<EMemoryType::GPU_Malloc> hashDag,
    gsl::span<const T> inItems, gsl::span<const uint32_t> indicesToUnique,
    gsl::span<uint32_t> inOutHandles,
    uint32_t* pNumOutItems, uint32_t* pOutItemsBytes)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();

    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < inItems.size() + warpsPerBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx >= inItems.size())
            return;

        // Unique in SVO and not already in DAG.
        if (indicesToUnique[warpIdx] == not_found_sentinel && inOutHandles[warpIdx] == hashDag.invalid_handle) {
            const uint32_t* pItem = &inItems[warpIdx].padding[0];
            if constexpr (level == hashDag.leaf_level()) {
                inOutHandles[warpIdx] = hashDag.add_leaf_as_warp(pItem);
            } else {
                inOutHandles[warpIdx] = hashDag.add_node_as_warp(pItem);
            }
#if CAPTURE_MEMORY_STATS_SLOW
            if (warp.thread_rank() == 0) {
                atomicAdd(pNumOutItems, 1);
                atomicAdd(pOutItemsBytes, inItems[warpIdx].size * sizeof(uint32_t));
            }
#endif
        }
    }
}

template <typename T>
__global__ void gatherHandlesOfDuplicateNodes(
    gsl::span<const uint32_t> indicesToUnique, gsl::span<const uint32_t> dagHandles, gsl::span<NodeInformation> outInfos)
{
    const unsigned globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIndex >= dagHandles.size())
        return;

    // Edit tool created an empty node.
    if (indicesToUnique[globalThreadIndex] == empty_sentinel) {
        auto& info = outInfos[globalThreadIndex];
        info.fullyFilledMaterial = sentinel;
        info.handle = sentinel;
        return;
    }

    uint32_t uniqueIndexInSvo = globalThreadIndex;
    uint32_t numIndirections = 0;
    uint32_t indirection;
    while ((indirection = indicesToUnique[uniqueIndexInSvo]) != not_found_sentinel) {
        uniqueIndexInSvo = indirection;
        ++numIndirections;
    }
    check(numIndirections <= 2);

    auto& info = outInfos[globalThreadIndex];
    info.fullyFilledMaterial = sentinel;
    info.handle = dagHandles[uniqueIndexInSvo];
}

template <uint32_t level>
static void doMergeDagHashing(
    IntermediateSVO& inSvo, MyGPUHashDAG<EMemoryType::GPU_Malloc>& outDag,
    gsl::span<const NodeInformation> inChildMapping, gsl::span<NodeInformation> outMapping, uint32_t& inOutHasFullyFilledNodes, uint32_t* pNumOutItems, uint32_t* pInItemsBytes, uint32_t* pOutItemsBytes,
    GpuMemoryPool& memPool, cudaStream_t stream, GPUTimingsManager& timingsManager)
{
    PROFILE_FUNCTION();

    auto inNodes = getData<level>(inSvo);
    using TItem = typename decltype(inNodes)::value_type;
    using TDAG = MyGPUHashDAG<EMemoryType::GPU_Malloc>;

    if constexpr (level != MyGPUHashDAG<EMemoryType::GPU_Malloc>::leaf_level()) {
        // Update child pointers.
        // constexpr auto headerSize = MyGPUHashDAG<EMemoryType::GPU_Malloc>::get_header_size(level);
        deviceSynchronizeTiming();
        PROFILE_SCOPE("updateChildPointers");
        const auto timing = timingsManager.timeScope("updateChildPointers", stream);
        updateChildPointers<<<computeNumWorkGroups(inNodes.size()), workGroupSize, 0, stream>>>(
            outDag, level,
            inNodes, inChildMapping, outMapping);
        deviceSynchronizeTiming();
        CUDA_CHECK_ERROR();
    }

    if (inOutHasFullyFilledNodes) {
        uint32_t* pHasFullyFilledNodes = memPool.mallocAsync<uint32_t>();
        cudaMemsetAsync(pHasFullyFilledNodes, 0, sizeof(uint32_t), stream);

        deviceSynchronizeTiming();
        PROFILE_SCOPE("discover_fully_filled");
        const auto timing = timingsManager.timeScope("discover_fully_filled", stream);
        discover_fully_filled<level, TItem><<<computeNumWorkGroups(inNodes.size()), workGroupSize, 0, stream>>>(
            inNodes, inChildMapping, outDag, outMapping, pHasFullyFilledNodes);
        CUDA_CHECK_ERROR();
        deviceSynchronizeTiming();

        cudaMemcpy(&inOutHasFullyFilledNodes, pHasFullyFilledNodes, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        memPool.freeAsync(pHasFullyFilledNodes);
    }

    // Remove duplicate nodes to prevent duplicate nodes in the final dag due to race conditions (between find & insert).
    const auto itemSizeInU32 = TItem::MaxSizeInU32;
    const uint32_t numItems = (uint32_t)inNodes.size();
    const uint32_t numBuckets = numItems / 32 + 1;
    // auto hashTable = Atomic64HashTable<EMemoryType::GPU_Async, true>::allocate(numBuckets, 4 * numItems, itemSizeInU32);
    auto hashTable = DuplicateDetectionHashTable::allocate(numBuckets, numItems + 10000, itemSizeInU32);

    CUDA_CHECK_ERROR();
    const uint32_t numWorkGroups = std::min(((uint32_t)inNodes.size() + warpsPerWorkGroup - 1) / warpsPerWorkGroup, maxWorkGroups);

    auto indicesToUniqueItemInSvo = memPool.mallocAsync<uint32_t>(inNodes.size());
    {
        const auto timing = timingsManager.timeScope("memset_duplicate_indices", stream);
        cudaMemsetAsync(indicesToUniqueItemInSvo.data(), 0xFFFFFFFF, indicesToUniqueItemInSvo.size_bytes(), stream);
    }
    {
        // Insert all items into the hash table. Multiple insertions of the same item will result in duplicates in the hash table.
        deviceSynchronizeTiming();
        PROFILE_SCOPE("insertDuplicatesIntoHashTableAsWarp_kernel");
        const auto timing = timingsManager.timeScope("insertDuplicatesIntoHashTableAsWarp_kernel", stream);
        insertDuplicatesIntoHashTableAsWarp_kernel<TItem, level == TDAG::leaf_level()><<<numWorkGroups, warpsPerWorkGroup * 32, 0, stream>>>(
            hashTable, inNodes, indicesToUniqueItemInSvo, pInItemsBytes);
        CUDA_CHECK_ERROR();
        deviceSynchronizeTiming();
    }

    auto hashTableHandles = memPool.mallocAsync<uint32_t>(inNodes.size());
    // Combines two steps:
    // 1. Find items in the hash table we just created. If the index that we get back is different then we are a duplicate and we stop.
    // 2. If we were the first unique instance then we search the SVDAG for a match.
    //
    // The result are stored in two arrays:
    // - indicesToUniqueItemInSvo: index to the unique instances (which have 0xFFFFFFFF); may contain 2 levels of indirection.
    // - hashTableHandles: for the unique instances (in the SVO), a hash table handle to a node in the SVDAG (if found).
    {
        PROFILE_SCOPE("findUniqueInHashTableAsWarp_kernel1");
        const auto timing = timingsManager.timeScope("findUniqueInHashTableAsWarp_kernel1", stream);
        findUniqueInHashTableAsWarp_kernel1<TItem, level><<<numWorkGroups, warpsPerWorkGroup * 32, 0, stream>>>(
            hashTable, outDag, inNodes, indicesToUniqueItemInSvo);
        CUDA_CHECK_ERROR();
        deviceSynchronizeTiming();
    }
    {
        PROFILE_SCOPE("findUniqueInHashTableAsWarp_kernel2");
        const auto timing = timingsManager.timeScope("findUniqueInHashTableAsWarp_kernel2", stream);
        findUniqueInHashTableAsWarp_kernel2<TItem, level><<<numWorkGroups, warpsPerWorkGroup * 32, 0, stream>>>(
            hashTable, outDag, inNodes, indicesToUniqueItemInSvo, hashTableHandles);
        CUDA_CHECK_ERROR();
        deviceSynchronizeTiming();
    }

    {
        PROFILE_SCOPE("hashTable.free()");
        const auto timing = timingsManager.timeScope("hashTable.free()", stream);
        hashTable.free();
        deviceSynchronizeTiming();
    }
    {
        // Insert the completely unique nodes (currently set to sentinel) into the SVDAG.
        deviceSynchronizeTiming();
        PROFILE_SCOPE("insertUniqueInHashTableAsWarp_kernel");
        const auto timing = timingsManager.timeScope("insertUniqueInHashTableAsWarp_kernel", stream);
        insertUniqueInHashTableAsWarp_kernel<TItem, level><<<numWorkGroups, warpsPerWorkGroup * 32, 0, stream>>>(
            outDag, inNodes, indicesToUniqueItemInSvo, hashTableHandles, pNumOutItems, pOutItemsBytes);
        CUDA_CHECK_ERROR();
        deviceSynchronizeTiming();
    }
    {
        // Copy the results from the unique (in SVO) items to the duplicate items.
        deviceSynchronizeTiming();
        PROFILE_SCOPE("gatherHandlesOfDuplicateNodes");
        const auto timing = timingsManager.timeScope("gatherHandlesOfDuplicateNodes", stream);
        gatherHandlesOfDuplicateNodes<TItem><<<computeNumWorkGroups(inNodes.size()), workGroupSize, 0, stream>>>(
            indicesToUniqueItemInSvo, hashTableHandles, outMapping);
        CUDA_CHECK_ERROR();
        deviceSynchronizeTiming();
    }
    memPool.freeAsync(indicesToUniqueItemInSvo);

    {
        deviceSynchronizeTiming();
        PROFILE_SCOPE("memPool.freeAsync(hashTableHandles);");
        const auto timing = timingsManager.timeScope("memPool.freeAsync(hashTableHandles);", stream);
        memPool.freeAsync(hashTableHandles);
        deviceSynchronizeTiming();
    }

#if ENABLE_CHECKS
    // Make sure that, after adding the new nodes, there are still no duplicate nodes in the DAG.
    outDag.checkHashTables();
#endif

    CUDA_CHECK_ERROR();
}

static __global__ void computeNodeSize_kernel(gsl::span<const IntermediateSVO::Node> inNodes, gsl::span<uint32_t> outNodeSizeHistogram)
{
    using TDAG = MyGPUHashDAG<EMemoryType::GPU_Malloc>;

    const unsigned globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t threadInWarp = threadIdx.x & 31;
    if (globalThreadIndex < inNodes.size()) {
        const auto& node = inNodes[globalThreadIndex];
        if (node.size == 0)
            return;

        check(TDAG::get_node_size(node.padding) == node.size);
        const uint32_t othersMask = __match_any_sync(0xFFFFFFFF, node.size);
        check(othersMask != 0);
        if (__ffs(othersMask) - 1u == threadInWarp)
            atomicAdd(&outNodeSizeHistogram[node.size], __popc(othersMask));
        // atomicAdd(&outNodeSizeHistogram[node.size], 1);
    }
}
static __global__ void computeLeafSize_kernel(gsl::span<const IntermediateSVO::Leaf> inLeaves, gsl::span<uint32_t> outLeafSizeHistogram)
{
    using TDAG = MyGPUHashDAG<EMemoryType::GPU_Malloc>;

    const unsigned globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t threadInWarp = threadIdx.x & 31;
    if (globalThreadIndex < inLeaves.size()) {
        const auto& leaf = inLeaves[globalThreadIndex];
        check(TDAG::get_leaf_size(leaf.padding) == leaf.size);
        const uint32_t othersMask = __match_any_sync(0xFFFFFFFF, leaf.size);
        check(othersMask != 0);
        if (__ffs(othersMask) - 1u == threadInWarp)
            atomicAdd(&outLeafSizeHistogram[leaf.size], __popc(othersMask));
        // atomicAdd(&outLeafSizeHistogram[leaf.size], 1);
    }
}

template <uint32_t level>
auto getData(IntermediateSVO& inSvo)
{
    return inSvo.innerNodes[level];
}
template <>
auto getData<BaseDAG::leaf_level()>(IntermediateSVO& inSvo)
{
    return inSvo.leaves;
}

static void mergeDag_CUDA(
    IntermediateSVO& inSvo, MyGPUHashDAG<EMemoryType::GPU_Malloc>& outDag,
    StatsRecorder& stats, GpuMemoryPool& memPool, cudaStream_t stream, GPUTimingsManager& timingsManager)
{
    PROFILE_FUNCTION();
    using TDAG = MyGPUHashDAG<EMemoryType::GPU_Malloc>;

    uint32_t innerNodeCount = 0, maxLevelSize = 0;
    for (const auto& nodes : inSvo.innerNodes) {
        innerNodeCount += (uint32_t)nodes.size();
        maxLevelSize = std::max(maxLevelSize, (uint32_t)nodes.size());
    }

    const uint32_t leafNodeCount = (uint32_t)inSvo.leaves.size();
    maxLevelSize = std::max(maxLevelSize, leafNodeCount);

    // Bottom-up insertion of the new nodes into the outDag
    gsl::span<NodeInformation> prevLevelMapping, curLevelMapping;
    {
        PROFILE_SCOPE("Allocate temporary memory");
        auto timings = timingsManager.timeScope("Allocate temporaries", stream);
        prevLevelMapping = memPool.mallocAsync<NodeInformation>(maxLevelSize);
        curLevelMapping = memPool.mallocAsync<NodeInformation>(maxLevelSize);
    }

    {
        std::vector<uint32_t> nodesPerSize, leavesPerSize;
        {
            PROFILE_SCOPE("Predict DAG allocator growth");
#if HASH_TABLE_ACCURATE_RESERVE_MEMORY
            auto timings = timingsManager.timeScope("predict_allocator_growth", stream);
            auto nodeSizeHistogramGPU = memPool.mallocAsync<uint32_t>(TDAG::maxItemSizeInU32 + 1);
            auto leafSizeHistogramGPU = memPool.mallocAsync<uint32_t>(TDAG::maxItemSizeInU32 + 1);
            cudaMemsetAsync(nodeSizeHistogramGPU.data(), 0, nodeSizeHistogramGPU.size_bytes(), stream);
            cudaMemsetAsync(leafSizeHistogramGPU.data(), 0, leafSizeHistogramGPU.size_bytes(), stream);
            for (const auto& levelNodes : inSvo.innerNodes) {
                if (levelNodes.size() == 0)
                    continue;
                computeNodeSize_kernel<<<computeNumWorkGroups(levelNodes.size()), workGroupSize, 0, stream>>>(levelNodes, nodeSizeHistogramGPU);
            }
            computeLeafSize_kernel<<<computeNumWorkGroups(inSvo.leaves.size()), workGroupSize, 0, stream>>>(inSvo.leaves, leafSizeHistogramGPU);

            nodesPerSize.resize(nodeSizeHistogramGPU.size());
            cudaMemcpyAsync(nodesPerSize.data(), nodeSizeHistogramGPU.data(), nodeSizeHistogramGPU.size_bytes(), cudaMemcpyDeviceToHost, stream);
            leavesPerSize.resize(leafSizeHistogramGPU.size());
            cudaMemcpyAsync(leavesPerSize.data(), leafSizeHistogramGPU.data(), leafSizeHistogramGPU.size_bytes(), cudaMemcpyDeviceToHost, stream);
            memPool.freeAsync(nodeSizeHistogramGPU.data());
            memPool.freeAsync(leafSizeHistogramGPU.data());
            // Wait for data to be copied to the CPU.
            cudaStreamSynchronize(stream);
#else
            nodesPerSize.resize(TDAG::maxItemSizeInU32 + 1, 0);
            for (uint32_t i = TDAG::minNodeSizeInU32; i <= TDAG::maxNodeSizeInU32; ++i)
                nodesPerSize[i] = innerNodeCount;
            leavesPerSize.resize(TDAG::maxItemSizeInU32 + 1, 0);
            for (uint32_t i = TDAG::minLeafSizeInU32; i <= TDAG::maxLeafSizeInU32; ++i)
                leavesPerSize[i] = leafNodeCount;
#endif
        }

        {
            PROFILE_SCOPE("Grow DAG allocators");
            auto timings = timingsManager.timeScope("Grow DAG allocators", stream);

            // Make sure that the DAG has enough free memory to store any new nodes.
            // Reserve more memory (and copy over the data) if this is not the case.
            std::vector<uint32_t> itemsPerSize(nodesPerSize.size());
            for (uint32_t i = 0; i < itemsPerSize.size(); ++i)
                itemsPerSize[i] = nodesPerSize[i] + leavesPerSize[i];
            CUDA_CHECK_ERROR();
            outDag.reserveIfNecessary(itemsPerSize);
        }
    }

    {
        PROFILE_SCOPE("merge DAGs");

        uint32_t numInItems = 0;
        uint32_t* pNumOutItems = nullptr;
        uint32_t* pInItemsBytes = nullptr;
        uint32_t* pOutItemsBytes = nullptr;
#if CAPTURE_MEMORY_STATS_SLOW
        pNumOutItems = memPool.mallocAsync<uint32_t>();
        pInItemsBytes = memPool.mallocAsync<uint32_t>();
        pOutItemsBytes = memPool.mallocAsync<uint32_t>();
        cudaMemsetAsync(pNumOutItems, 0, sizeof(uint32_t), stream);
        cudaMemsetAsync(pInItemsBytes, 0, sizeof(uint32_t), stream);
        cudaMemsetAsync(pOutItemsBytes, 0, sizeof(uint32_t), stream);
#endif

#if ENABLE_CHECKS
        outDag.checkHashTables();
#endif

        uint32_t hasFullyFilledNodes = true;
        Utils::constexpr_for_loop<TDAG::leaf_level() + 1>(
            [&](auto invLevel) {
                constexpr auto level = TDAG::leaf_level() - decltype(invLevel)::v;
                numInItems += (uint32_t)getData<level>(inSvo).size();
                doMergeDagHashing<level>(inSvo, outDag, prevLevelMapping, curLevelMapping, hasFullyFilledNodes, pNumOutItems, pInItemsBytes, pOutItemsBytes, memPool, stream, timingsManager);

                std::swap(prevLevelMapping, curLevelMapping);
            });
        /*{
            PROFILE_SCOPE("WaitForIdle");
            const auto timing = timingsManager.timeScope("waitForIdle", stream);
            cudaDeviceSynchronize(); // Required! Will crash otherwise!
        }*/
        CUDA_CHECK_ERROR();

#if CAPTURE_MEMORY_STATS_SLOW
        uint32_t numOutItems, inItemsBytes, outItemsBytes;
        cudaMemcpy(&numOutItems, pNumOutItems, sizeof(numOutItems), cudaMemcpyDeviceToHost);
        cudaMemcpy(&inItemsBytes, pInItemsBytes, sizeof(numOutItems), cudaMemcpyDeviceToHost);
        cudaMemcpy(&outItemsBytes, pOutItemsBytes, sizeof(numOutItems), cudaMemcpyDeviceToHost);
        memPool.freeAsync(pOutItemsBytes);
        memPool.freeAsync(pInItemsBytes);
        memPool.freeAsync(pNumOutItems);

        stats.reportInt("num_items_edit_svo", numInItems, "items", Device::GPU);
        stats.reportInt("num_unique_items_edit_svo", numOutItems, "items", Device::GPU);
        stats.report("edit_svo_size_bytes", (my_units::bytes)inItemsBytes, Device::GPU);
        stats.report("edit_insert_size_bytes", (my_units::bytes)outItemsBytes, Device::GPU);

        printf("Merging SVO with %u items (%.1f KB) added %u unique items (%.1f KB) to the DAG\n", numInItems, Utils::to_KB(inItemsBytes), numOutItems, Utils::to_KB(outItemsBytes));
#endif
    }

    {
        PROFILE_SCOPE("Cleanup");
        auto timings = timingsManager.timeScope("Cleanup", stream);
        cudaMemcpy(&outDag.firstNodeIndex, prevLevelMapping.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost);
        check(outDag.firstNodeIndex != outDag.invalid_handle);
        memPool.freeAsync(prevLevelMapping);
        memPool.freeAsync(curLevelMapping);
    }
}

template <typename Editor>
void editMyHashDag(const Editor& editor, MyGPUHashDAG<EMemoryType::GPU_Malloc>& dag, MyGPUHashDAGUndoRedo& undoRedo, StatsRecorder& statsRecorder, GpuMemoryPool& memPool, cudaStream_t stream)
{
    PROFILE_FUNCTION();

#ifdef UNDO_REDO
    undoRedo.add_frame({ .firstNodeIndex = dag.get_first_node_index() });
#endif

    GPUTimingsManager timingsManager {};
    timingsManager.startTiming("editing", stream);

    BasicStats stats;
    stats.start_work("[CPU] create_edit_svo_gpu");
    auto intermediateSVO = create_edit_intermediate_svo(dag, editor, memPool, stream, timingsManager);
    deviceSynchronizeTiming();
    stats.flush(statsRecorder);

    if (intermediateSVO.leaves.size() != 0) {
        stats.start_work("[CPU] sortAndMerge_CUDA");
        mergeDag_CUDA(intermediateSVO, dag, statsRecorder, memPool, stream, timingsManager);
        stats.flush(statsRecorder);
    }

    stats.start_work("[CPU] intermediateSVO.free()");
    {
        PROFILE_SCOPE("intermediateSVO.free");
        auto timings = timingsManager.timeScope("intermediateSVO.free()", stream);
        intermediateSVO.free(memPool);
#if 0
        cudaDeviceSynchronize();
#endif
    }
    stats.flush(statsRecorder);
    timingsManager.endTiming("editing", stream);

    timingsManager.flush(statsRecorder);
    timingsManager.print();
}

#define TOOL_IMPL(Tool) template void editMyHashDag(const Tool&, MyGPUHashDAG<EMemoryType::GPU_Malloc>&, MyGPUHashDAGUndoRedo&, StatsRecorder&, GpuMemoryPool&, cudaStream_t);

TOOL_IMPL(MyGpuBoxEditor<true>)
TOOL_IMPL(MyGpuBoxEditor<false>)
TOOL_IMPL(MyGpuSphereEditor<true>)
TOOL_IMPL(MyGpuSphereEditor<false>)
TOOL_IMPL(MyGpuSpherePaintEditor)
TOOL_IMPL(MyGpuSphereErosionEditor<MyGPUHashDAG<EMemoryType::GPU_Malloc>>)
TOOL_IMPL(MyGpuCopyEditor<MyGPUHashDAG<EMemoryType::GPU_Malloc>>)
