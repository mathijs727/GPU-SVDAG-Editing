#include "cuda_helpers.h"
#include "dags/my_gpu_dags/cub/cub_radix_sort.h"
#include "dags/my_gpu_dags/cub/cub_scan.h"
#include "dags/my_gpu_dags/cub/cub_select.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "my_units.h"
#include "stats.h"
#include "utils.h"
#include <algorithm>
#include <compare>
#include <cuda.h>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_map>

template <EMemoryType memoryType>
MyGPUHashDAG<memoryType> MyGPUHashDAG<memoryType>::allocate(uint32_t hashTableSize)
{
    std::vector<uint32_t> tableSizes(MyGPUHashDAG::maxItemSizeInU32 - MyGPUHashDAG::minItemSizeInU32 + 1, hashTableSize);
    return allocate(tableSizes);
}

template <EMemoryType memoryType>
MyGPUHashDAG<memoryType> MyGPUHashDAG<memoryType>::allocate(gsl::span<const uint32_t> tableSizes)
{
    MyGPUHashDAG out;
    out.hashTables = StaticArray<HashTableImpl>::allocate(
        "MyGPUHashDAG::hashTables", Super::maxItemSizeInU32 - Super::minItemSizeInU32 + 1, memoryType == EMemoryType::CPU ? EMemoryType::CPU : EMemoryType::GPU_Managed);
    for (uint32_t i = 0; i < out.hashTables.size(); ++i) {
        out.hashTables[i] = HashTableImpl::allocate(tableSizes[i], 1u, out.minItemSizeInU32 + i);
    }

    if constexpr (memoryType != EMemoryType::CPU) {
        cudaMemAdvise(out.hashTables.data(), out.hashTables.size_in_bytes(), cudaMemAdviseSetReadMostly, 0);
    }

    // Insert fully filled leaves for each material type.
    out.fullyFilledNodes = StaticArray2D<uint32_t>::allocate("MyGPUHashDAG::fullyFilledNodes", MAX_LEVELS, NumMaterials, memoryType);
    for (uint32_t material = 0; material < NumMaterials; ++material) {
        std::array<uint32_t, maxLeafSizeInU32> leafBuffer;
        LeafBuilder builder { leafBuffer.data() };
        for (uint32_t i = 0; i < 64; ++i) {
            builder.set(material);
            builder.next();
        }
        builder.finalize();
        out.fullyFilledNodes.write(MyGPUHashDAG::leaf_level(), material, out.add_leaf(leafBuffer.data()));
    }
    // Insert fully filled nodes for each material type.
    for (int level = MyGPUHashDAG::leaf_level() - 1; level >= 0; --level) {
        for (uint32_t material = 0; material < NumMaterials; ++material) {
            uint32_t nodeBuffer[maxNodeSizeInU32];
            uint32_t* pChildren = MyGPUHashDAG::encode_node_header(nodeBuffer, 0xFF, material);
            for (uint32_t i = 0; i < 8; ++i)
                pChildren[i] = out.fullyFilledNodes.read(level + 1, material);

            uint32_t nodeHandle = out.find_node(nodeBuffer);
            if (nodeHandle == MyGPUHashDAG::invalid_handle)
                nodeHandle = out.add_node(nodeBuffer);
            out.fullyFilledNodes.write(level, material, nodeHandle);
        }
    }

    return out;
}

template <EMemoryType memoryType>
void MyGPUHashDAG<memoryType>::free()
{
    PROFILE_FUNCTION();
    for (auto& hashTable : hashTables.copy_to_cpu())
        hashTable.free();
    hashTables.free();
    fullyFilledNodes.free();
}

struct PairHash {
    size_t operator()(const std::pair<uint32_t, uint32_t>& lhs) const
    {
        size_t s = 0;
        Utils::hash_combine_cpu(s, lhs.first);
        Utils::hash_combine_cpu(s, lhs.second);
        return s;
    }
};
struct CopyDagCache {
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, PairHash> nodes;
    std::unordered_map<uint32_t, uint32_t> leaves;
};
template <EMemoryType memoryType>
static uint32_t copyDag(const MyGPUHashDAG<memoryType>& inDag, CopyDagCache& cache, uint32_t level, uint32_t index, MyGPUHashDAG<memoryType>& outDag)
{
    if (level == inDag.leaf_level()) {
        if (auto iter = cache.leaves.find(index); iter != std::end(cache.leaves))
            return iter->second;

        const auto pInLeaf = inDag.get_leaf_ptr(index);
        std::array<uint32_t, MyGPUHashDAG<memoryType>::maxLeafSizeInU32> outLeaf;
        for (uint32_t i = 0; i < inDag.get_leaf_size(pInLeaf); ++i)
            outLeaf[i] = pInLeaf[i];
        const auto leafHandle = outDag.add_leaf(outLeaf.data());
        cache.leaves[index] = leafHandle;
        return leafHandle;
    } else {
        if (auto iter = cache.nodes.find({ level, index }); iter != std::end(cache.nodes))
            return iter->second;

        // Copy header.
        std::array<uint32_t, inDag.maxNodeSizeInU32> outNode;
        const auto pInNode = inDag.get_node_ptr(level, index);
        const auto headerSize = inDag.get_header_size();
        for (uint32_t i = 0; i < headerSize; ++i) {
            outNode[i] = pInNode[i];
        }
        // Move children into new DAG.
        const auto childCount = Utils::popc(Utils::child_mask(pInNode[0]));
        for (uint8_t childOffset = 0; childOffset < childCount; ++childOffset) {
            const auto inChild = inDag.get_child_index(level, index, childOffset);
            outNode[headerSize + childOffset] = copyDag(inDag, cache, level + 1, inChild, outDag);
        }

        const auto nodeHandle = outDag.add_node(outNode.data());
        cache.nodes[{ level, index }] = nodeHandle;
        return nodeHandle;
    }
}

template <EMemoryType memoryType>
double MyGPUHashDAG<memoryType>::getAverageLoadFactor() const
{
    double loadFactor = 0.0;
    for (const auto& hashTable : hashTables)
        loadFactor += hashTable.currentLoadFactor();
    return loadFactor / (double)hashTables.size();
}

template <EMemoryType memoryType>
void MyGPUHashDAG<memoryType>::setLoadFactor(double targetLoadFactor)
{
    checkAlways(memoryType == EMemoryType::CPU);

    std::vector<uint32_t> tableSizes;
    for (const auto& hashTable : hashTables) {
        const uint32_t targetNumBuckets = std::max(1u, (uint32_t)(hashTable.numBuckets() * hashTable.currentLoadFactor() / targetLoadFactor) + 1u);
        tableSizes.push_back(targetNumBuckets);
    }
    // Resizing the table will invalidate pointers.
    // Perform a traversal of the DAG and update all pointers.
    auto newDag = MyGPUHashDAG<memoryType>::allocate(tableSizes);
    CopyDagCache copyDagCache;
    newDag.firstNodeIndex = copyDag(*this, copyDagCache, 0, firstNodeIndex, newDag);
    this->free();
    *this = newDag;
}

template <EMemoryType memoryType>
void MyGPUHashDAG<memoryType>::writeTo(BinaryWriter& writer) const
{
    writer.write(version);

    writer.write(fullyFilledNodes);
    writer.write(firstNodeIndex);
    writer.write(hashTables);
}

template <EMemoryType memoryType>
void MyGPUHashDAG<memoryType>::readFrom(BinaryReader& reader)
{
    uint32_t controlVersion;
    reader.read(controlVersion);
    checkAlways(controlVersion == version);

    reader.read(fullyFilledNodes);
    reader.read(firstNodeIndex);
    reader.read(hashTables);
}

template <EMemoryType memoryType>
my_units::bytes MyGPUHashDAG<memoryType>::memory_allocated() const
{
    my_units::bytes out { 0 };
#if CAPTURE_MEMORY_STATS_SLOW
    // Requires CPU/GPU synchronization so only run this function when necessary.
    for (const auto& hashTable : hashTables.copy_to_cpu())
        out += hashTable.memory_allocated();

#endif // ENABLE_CHECKS
    return out;
}

template <EMemoryType memoryType>
my_units::bytes MyGPUHashDAG<memoryType>::memory_used_by_items() const
{
    my_units::bytes out { 0 };
#if CAPTURE_MEMORY_STATS_SLOW
    // Requires CPU/GPU synchronization so only run this function when necessary.
    for (const auto& hashTable : hashTables.copy_to_cpu())
        out += hashTable.memory_used_by_items();

#endif // ENABLE_CHECKS
    return out;
}

template <EMemoryType memoryType>
my_units::bytes MyGPUHashDAG<memoryType>::memory_used_by_slabs() const
{
    my_units::bytes out { 0 };
#if CAPTURE_MEMORY_STATS_SLOW
    // Requires CPU/GPU synchronization so only run this function when necessary.
    for (const auto& hashTable : hashTables.copy_to_cpu())
        out += hashTable.memory_used_by_slabs();
#endif // CAPTURE_MEMORY_STATS_SLOW
    return out;
}

template <EMemoryType memoryType>
void MyGPUHashDAG<memoryType>::report(StatsRecorder& statsRecorder) const
{
    const auto device = memoryType == EMemoryType::CPU ? Device::CPU : Device::GPU;

    statsRecorder.reportInt("MyGPUHashDAG.version", version, "version", device);
    statsRecorder.report("MyGPUHashDAG.memory_allocated", memory_allocated(), device);
    statsRecorder.report("MyGPUHashDAG.memory_used_by_items", memory_used_by_items(), device);
    statsRecorder.report("MyGPUHashDAG.memory_used_by_slabs", memory_used_by_slabs(), device);
    /* for (uint32_t i = 0; i < hashTables.size(); ++i) {
        statsRecorder.reportInt("MyGPUHashDAG.hashTables[i].numBuckets", hashTables[i].numBuckets(), "buckets", device);
        statsRecorder.reportFloat("MyGPUHashDAG.hashTables[i].currentLoadFactor", hashTables[i].currentLoadFactor(), "factor", device);
    }*/
}

template <EMemoryType memoryType>
void MyGPUHashDAG<memoryType>::reserveIfNecessary(gsl::span<const uint32_t> numNewItems)
{
    auto hashTablesCPU = hashTables.copy_to_cpu();
    CUDA_CHECK_ERROR();
    for (uint32_t itemSizeInU32 = this->minItemSizeInU32; itemSizeInU32 <= this->maxItemSizeInU32; ++itemSizeInU32) {
        if (numNewItems[itemSizeInU32] == 0)
            continue;

        CUDA_CHECK_ERROR();
        hashTablesCPU[itemSizeInU32 - this->minItemSizeInU32].reserveIfNecessary(numNewItems[itemSizeInU32]);
        CUDA_CHECK_ERROR();
    }
    CUDA_CHECK_ERROR();
    hashTables.upload_to_gpu(hashTablesCPU);
    CUDA_CHECK_ERROR();
}

template <EMemoryType memoryType>
static __global__ void countNodeChildCount(MyGPUHashDAG<memoryType> dag, uint32_t level, gsl::span<const uint32_t> inNodes, gsl::span<uint32_t> outChildCounts)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < inNodes.size()) {
        const uint32_t nodeHandle = inNodes[globalThreadIdx];
        const uint32_t childMask = Utils::child_mask(dag.get_node(level, nodeHandle));
        outChildCounts[globalThreadIdx] = Utils::popc(childMask);
    }
}

template <EMemoryType memoryType>
static __global__ void markNodesAsActive_kernel(MyGPUHashDAG<memoryType> dag, uint32_t level, gsl::span<const uint32_t> inNodes, gsl::span<uint32_t> inOutStart, gsl::span<uint32_t> outNodes)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < inNodes.size()) {
        const uint32_t nodeHandle = inNodes[globalThreadIdx];
        dag.markNodeAsActive(nodeHandle);

        const uint32_t childMask = Utils::child_mask(dag.get_node(level, nodeHandle));
        uint32_t outAddress = inOutStart[globalThreadIdx];
        for (uint8_t i = 0; i < 8; ++i) {
            if (childMask & (1u << i)) {
                outNodes[outAddress++] = dag.get_child_index(level, nodeHandle, childMask, i);
            }
        }
    }
}
template <EMemoryType memoryType>
static __global__ void markNodesAsActive_kernel(MyGPUHashDAG<memoryType> dag, uint32_t level, gsl::span<const uint32_t> inNodes, gsl::span<uint32_t> outNodes, uint32_t* pOutIndex)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < inNodes.size()) {
        const uint32_t nodeHandle = inNodes[globalThreadIdx];
        dag.markNodeAsActive(nodeHandle);

        const uint32_t childMask = Utils::child_mask(dag.get_node(level, nodeHandle));
        // uint32_t outAddress = atomicAdd(pOutIndex, Utils::popc(childMask));
        for (uint8_t i = 0; i < 8; ++i) {
            if (childMask & (1u << i)) {
                uint32_t outAddress = atomicAdd(pOutIndex, 1);
                outNodes[outAddress] = dag.get_child_index(level, nodeHandle, childMask, i);
            }
        }
    }
}
template <EMemoryType memoryType>
static __global__ void markLeavesAsActive_kernel(MyGPUHashDAG<memoryType> dag, gsl::span<const uint32_t> inNodes)
{
    const unsigned globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx < inNodes.size()) {
        dag.markLeafAsActive(inNodes[globalThreadIdx]);
    }
}

template <>
void MyGPUHashDAG<EMemoryType::CPU>::garbageCollect(gsl::span<const uint32_t> activeRootNodes)
{
    for (auto table : hashTables)
        table.clearActiveFlags();

    struct StackItem {
        uint32_t level;
        uint32_t handle;

        // constexpr bool operator<(const StackItem&) const = default;
        constexpr bool operator<(const StackItem& other) const
        {
            if (level < other.level)
                return true;
            else if (level > other.level)
                return false;
            else {
                return handle < other.handle;
            }
        }
        constexpr bool operator==(const StackItem&) const = default;
    };
    std::stack<StackItem> traversalStack;
    for (uint32_t handle : activeRootNodes)
        traversalStack.push({ .level = 0, .handle = handle });

    std::set<StackItem> traversedItems;
    while (!traversalStack.empty()) {
        const auto stackItem = traversalStack.top();
        traversalStack.pop();

        // Only traverse a node once.
        if (traversedItems.find(stackItem) != std::end(traversedItems))
            continue;
        traversedItems.insert(stackItem);

        if (stackItem.level == leaf_level()) {
            markLeafAsActive(stackItem.handle);
        } else {
            markNodeAsActive(stackItem.handle);

            const uint32_t childMask = Utils::child_mask(get_node(stackItem.level, stackItem.handle));
            for (uint8_t i = 0; i < 8; ++i) {
                if (childMask & (1u << i)) {
                    traversalStack.push({ .level = stackItem.level + 1, .handle = get_child_index(stackItem.level, stackItem.handle, childMask, i) });
                }
            }
        }
    }

    uint32_t numBytesFreed = 0;
    for (auto& hashTable : hashTables) {
        const auto numItemsFreed = hashTable.freeInactiveItems();
        numBytesFreed += numItemsFreed * hashTable.itemSizeInU32 * sizeof(uint32_t);
    }
    printf("Garbage collect freed %u KB\n", numBytesFreed >> 10);
}

template <>
void MyGPUHashDAG<EMemoryType::GPU_Malloc>::garbageCollect(gsl::span<const uint32_t> activeRootNodes)
{
    for (auto table : hashTables.copy_to_cpu())
        table.clearActiveFlags();

    bool currentRootPresent = false;
    for (uint32_t rootNode : activeRootNodes)
        currentRootPresent |= (rootNode == get_first_node_index());
    checkAlways(currentRootPresent); // Make sure we are not deleting the current root node.

    cudaStream_t stream = nullptr;
    auto memoryPool = GpuMemoryPool::create(stream, EMemoryType::GPU_Malloc);
    auto inWorkItems = memoryPool.mallocAsync<uint32_t>(activeRootNodes.size() + fullyFilledNodes.height);
    cudaMemcpyAsync(inWorkItems.data(), activeRootNodes.data(), activeRootNodes.size_bytes(), cudaMemcpyHostToDevice, stream);
    for (int i = 0; i < fullyFilledNodes.height; ++i)
        cudaMemcpyAsync(&inWorkItems[1 + i], fullyFilledNodes.getPixelPointer(0, i), sizeof(uint32_t), cudaMemcpyDefault, stream);

    for (uint32_t level = 0; level < this->leaf_level(); ++level) {
        check(inWorkItems.size() > 0);

#if 1
        CUDA_CHECK_ERROR();
        auto outLocations = memoryPool.mallocAsync<uint32_t>(inWorkItems.size() + 1);
        cudaMemsetAsync(&outLocations.back(), 0, sizeof(uint32_t), stream);
        CUDA_CHECK_ERROR();

        countNodeChildCount<<<computeNumWorkGroups(inWorkItems.size()), workGroupSize, 0, stream>>>(
            *this, level, inWorkItems, outLocations);
        CUDA_CHECK_ERROR();

        cubExclusiveSum<uint32_t>(outLocations, outLocations, stream);
        CUDA_CHECK_ERROR();

        uint32_t outputSizeWithDuplicates;
        cudaMemcpyAsync(&outputSizeWithDuplicates, &outLocations.back(), sizeof(outputSizeWithDuplicates), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        CUDA_CHECK_ERROR();

        auto outWorkItems = memoryPool.mallocAsync<uint32_t>(outputSizeWithDuplicates);
        CUDA_CHECK_ERROR();

        markNodesAsActive_kernel<<<computeNumWorkGroups(inWorkItems.size()), workGroupSize, 0, stream>>>(
            *this, level, inWorkItems, outLocations, outWorkItems);
        CUDA_CHECK_ERROR();

        memoryPool.freeAsync(outLocations);
#else
        // Allocate space for the array of handles to the next level.
        CUDA_CHECK_ERROR();
        auto outWorkItems = memoryPool.mallocAsync<uint32_t>(inWorkItems.size() * 8);
        CUDA_CHECK_ERROR();

        // Mark all nodes in inWorkItems and store all their child pointers in outWorkItems.
        uint32_t* pOutputIndex = memoryPool.mallocAsync<uint32_t>();
        cudaMemsetAsync(pOutputIndex, 0, sizeof(uint32_t), stream);
        markNodesAsActive_kernel<<<computeNumWorkGroups(inWorkItems.size()), workGroupSize, 0, stream>>>(*this, level, inWorkItems, outWorkItems, pOutputIndex);
        CUDA_CHECK_ERROR();

        // Read how many (possibly duplicate) child handles have been written and resize the outWorkItems array accordingly.
        uint32_t outputSizeWithDuplicates;
        cudaMemcpyAsync(&outputSizeWithDuplicates, pOutputIndex, sizeof(outputSizeWithDuplicates), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        memoryPool.freeAsync(pOutputIndex);
        outWorkItems = outWorkItems.subspan(0, outputSizeWithDuplicates);
        CUDA_CHECK_ERROR();
#endif

        // Remove duplicates in outWorkItems so that we don't waste time/memory on visiting nodes/subtrees multiple times.
        auto sortedWorkItems = memoryPool.mallocAsync<uint32_t>(outWorkItems.size());
        cubDeviceRadixSortKeys<uint32_t>(outWorkItems, sortedWorkItems, stream);
        CUDA_CHECK_ERROR();
        uint32_t* pNumUniqueWorkItems = memoryPool.mallocAsync<uint32_t>();
        cudaMemsetAsync(pNumUniqueWorkItems, 0, sizeof(uint32_t), stream);
        cubDeviceSelectUnique<uint32_t>(sortedWorkItems, outWorkItems, pNumUniqueWorkItems, stream);
        CUDA_CHECK_ERROR();
        memoryPool.freeAsync(sortedWorkItems);
        CUDA_CHECK_ERROR();

        // Read how many unique child handles were produced and resize the output array accordingly.
        uint32_t outputSizeWithoutDuplicates;
        cudaMemcpyAsync(&outputSizeWithoutDuplicates, pNumUniqueWorkItems, sizeof(outputSizeWithoutDuplicates), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        outWorkItems = outWorkItems.subspan(0, outputSizeWithoutDuplicates);
        memoryPool.freeAsync(pNumUniqueWorkItems);
        CUDA_CHECK_ERROR();

        // Swap input/output for the next loop iteration.
        memoryPool.freeAsync(inWorkItems);
        inWorkItems = outWorkItems;
        CUDA_CHECK_ERROR();
    }

    // Mark all leaves in inWorkItems.
    markLeavesAsActive_kernel<<<computeNumWorkGroups(inWorkItems.size()), workGroupSize, 0, stream>>>(*this, inWorkItems);
    CUDA_CHECK_ERROR();

    memoryPool.freeAsync(inWorkItems);
    cudaStreamSynchronize(stream);
    CUDA_CHECK_ERROR();

#if CAPTURE_MEMORY_STATS_SLOW
    uint32_t numBytesFreed = 0;
    for (auto& hashTable : hashTables.copy_to_cpu()) {
        const auto numItemsFreed = hashTable.freeInactiveItems();
        numBytesFreed += numItemsFreed * hashTable.itemSizeInU32 * sizeof(uint32_t);
    }
    printf("Garbage collect freed %u KB\n", numBytesFreed >> 10);
#else
    printf("Enable CAPTURE_MEMORY_STATS_SLOW to report garbage collection freed memory\n");
#endif
}

template struct MyGPUHashDAG<EMemoryType::CPU>;
template struct MyGPUHashDAG<EMemoryType::GPU_Malloc>;
