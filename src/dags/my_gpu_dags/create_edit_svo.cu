#include "safe_cooperative_groups.h"
#include <cooperative_groups/reduce.h>
#include <cuda.h>
// ^^^ INCLUDE FIRST ^^^

#include "create_edit_svo.h"

#include "configuration/gpu_hash_dag_definitions.h"
#include "cuda_helpers.h"
#include "cuda_helpers_cpp.h"
#include "dags/my_gpu_dags/my_gpu_dag_editors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "stats.h"
#include "timings.h"
#include "work_queue.h"
#include <deque>
#include <iostream>
#include <tbb/parallel_for.h>

#define FILL_FULL_REGION 1
#define TIMING_SYNCHRONIZATION 0

static void deviceSynchronizeTiming()
{
#if TIMING_SYNCHRONIZATION
    cudaDeviceSynchronize();
#endif
}

void IntermediateSVO::free(GpuMemoryPool& memPool)
{
    PROFILE_FUNCTION();
    for (auto& nodesLevel : innerNodes)
        memPool.freeAsync(nodesLevel);
    memPool.freeAsync(leaves);
}

size_t IntermediateSVO::size_in_bytes() const
{
    size_t out = sizeof(*this);
    out += leaves.size_bytes();
    for (const auto& level : innerNodes)
        out += level.size_bytes();
    return out;
}

size_t IntermediateSVO::largestLevel() const
{
    size_t out = leaves.size();
    for (const auto& levelNodes : innerNodes)
        out = std::max(out, levelNodes.size());
    return out;
}

template <typename EditorState>
struct CreateIntermediateWorkItem {
    Path path;
    uint32_t dagNodeHandle;
    EditorState editorState;
};
template <>
struct CreateIntermediateWorkItem<void> {
    Path path;
    uint32_t dagNodeHandle;
};
template <typename DAG, typename Editor, typename InEditorState, typename OutEditorState>
static __global__ void createIntermediateSvoStructure_innerNode_cuda(
    const DAG inDag, const Editor editor, uint32_t level,
    gsl::span<const CreateIntermediateWorkItem<InEditorState>> inWorkItems, gsl::span<typename IntermediateSVO::Node> outNodes,
    gsl::span<CreateIntermediateWorkItem<OutEditorState>> outWorkItems, uint32_t* pIndex)
{
    const auto globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= inWorkItems.size())
        return;

    const uint32_t childLevel = level + 1;
    const auto childDepth = C_maxNumberOfLevels - childLevel;

    auto inWorkItem = inWorkItems[globalThreadIdx];
    typename DAG::NodeDecoder pOldNode;
    uint32_t oldBitmask = 0;
    if (inWorkItem.dagNodeHandle != sentinel) {
        pOldNode = inDag.get_node_ptr(level, inWorkItem.dagNodeHandle);
        oldBitmask = Utils::child_mask(pOldNode[0]);
    }

    // Traverse the to-be-edited region (except for fully filled) and create nodes & leafs in the SVO.
    auto& outNode = outNodes[globalThreadIdx];
    uint32_t childIntermediateMask = 0;
    uint32_t childMask = 0, childOffset = 0;
    outNode.size = inDag.get_header_size();
    for (uint32_t childIdx = 0; childIdx < 8; ++childIdx) {
        Path childPath = inWorkItem.path;
        childPath.descend(childIdx);

        uint32_t childHandle;
        NodeDesc nodeDesc;
        // Load handle to child if that child node exists in the original DAG.
        if (oldBitmask & (1u << childIdx)) {
            childHandle = inDag.get_child_index(level, pOldNode, oldBitmask, childIdx);
            if (childLevel == inDag.leaf_level()) {
                nodeDesc.state = NodeState::Partial;
            } else {
                nodeDesc.state = inDag.get_node_is_fully_filled(*inDag.get_node_ptr(childLevel, childHandle), nodeDesc.fillMaterial) ? NodeState::Filled : NodeState::Partial;
            }
        } else {
            childHandle = sentinel;
            nodeDesc.state = NodeState::Empty;
        }

        if constexpr (std::is_same_v<InEditorState, void>) {
            NodeEditDesc nodeEditDesc = editor.should_edit(childPath, childDepth, nodeDesc);

#if !EDITS_ENABLE_MATERIALS
            nodeEditDesc.fillMaterial = 0;
#endif

#if !FILL_FULL_REGION
            if (nodeEditDesc.operation == EditOperation::Fill || nodeEditDesc.operation == EditOperation::Empty)
                nodeEditDesc.operation = EditOperation::Subdivide;
#endif

            if (nodeEditDesc.operation == EditOperation::Subdivide) {
                const uint32_t outIndex = atomicAdd(pIndex, 1);
                outWorkItems[outIndex] = { childPath, childHandle };
                childHandle = outIndex;
                childIntermediateMask |= (1u << childIdx);
            } else if (nodeEditDesc.operation == EditOperation::Empty) {
                childHandle = sentinel;
            } else if (nodeEditDesc.operation == EditOperation::Fill) {
                childHandle = inDag.fullyFilledNodes.read(childLevel, nodeEditDesc.fillMaterial);
            }
        } else {
            NodeEditDesc nodeEditDesc;
            OutEditorState childEditorState;
            nodeEditDesc = editor.should_edit(childPath, childDepth, nodeDesc, inWorkItem.editorState, childEditorState);

#if !FILL_FULL_REGION
            if (nodeEditDesc.operation == EditOperation::Fill || nodeEditDesc.operation == EditOperation::Empty)
                nodeEditDesc.operation = EditOperation::Subdivide;
#endif

            if (nodeEditDesc.operation == EditOperation::Subdivide) {
                const uint32_t outIndex = atomicAdd(pIndex, 1);
                if constexpr (std::is_same_v<OutEditorState, void>)
                    outWorkItems[outIndex] = { childPath, childHandle };
                else
                    outWorkItems[outIndex] = { childPath, childHandle, childEditorState };
                childHandle = outIndex;
                childIntermediateMask |= (1u << childIdx);
            } else if (nodeEditDesc.operation == EditOperation::Empty) {
                childHandle = sentinel;
            } else if (nodeEditDesc.operation == EditOperation::Fill) {
                childHandle = inDag.fullyFilledNodes.read(childLevel, nodeEditDesc.fillMaterial);
            }
        }

        if (childHandle != sentinel) {
            outNode.padding[DAG::get_header_size() + childOffset++] = childHandle;
            childMask |= (1u << childIdx);
            ++outNode.size;
        }
    }

    DAG::encode_node_header_with_payload(&outNode.padding[0], childMask, childIntermediateMask);
}
template <typename DAG, typename Editor, typename InEditorState, typename OutEditorState>
static __global__ void createIntermediateSvoStructure_innerNode_warp(
    const DAG inDag, const Editor editor, uint32_t level,
    gsl::span<const CreateIntermediateWorkItem<InEditorState>> inWorkItems, gsl::span<typename IntermediateSVO::Node> outNodes,
    gsl::span<CreateIntermediateWorkItem<OutEditorState>> outWorkItems, uint32_t* pIndex)
{
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto nodeGroup = cooperative_groups::tiled_partition<8>(cooperative_groups::this_thread_block());

    const uint32_t nodeGroupsPerWorkGroup = nodeGroup.meta_group_size();
    const uint32_t nodeGroupInWorkGroup = nodeGroup.meta_group_rank();
    const uint32_t threadInNodeGroup = nodeGroup.thread_rank();
    const uint32_t nodeGroupThreadMask = 1u << threadInNodeGroup;
    const uint32_t nodeGroupThreadPrefixMask = nodeGroupThreadMask - 1u;

    const uint32_t threadInWarp = warp.thread_rank();
    const uint32_t warpThreadMask = 1u << threadInWarp;
    const uint32_t warpThreadPrefixMask = warpThreadMask - 1u;

    const uint32_t childLevel = level + 1;
    const auto childDepth = C_maxNumberOfLevels - childLevel;

    for (uint32_t firstNodeGroupIdx = blockIdx.x * nodeGroupsPerWorkGroup; firstNodeGroupIdx < inWorkItems.size() + nodeGroupsPerWorkGroup; firstNodeGroupIdx += gridDim.x * nodeGroupsPerWorkGroup) {
        const uint32_t nodeGroupIdx = firstNodeGroupIdx + nodeGroupInWorkGroup;
        if (nodeGroupIdx >= inWorkItems.size())
            return;

        auto inWorkItem = inWorkItems[nodeGroupIdx];
        auto& outNode = outNodes[nodeGroupIdx];

        // Load existing node if existed
        uint32_t childHandle = sentinel;
        if (inWorkItem.dagNodeHandle != sentinel) {
            const auto pOldNode = inDag.get_node_ptr(level, inWorkItem.dagNodeHandle);
            const uint32_t childMask = Utils::child_mask(pOldNode[0]);
            if (childMask & nodeGroupThreadMask)
                childHandle = pOldNode[DAG::get_header_size() + Utils::popc(childMask & nodeGroupThreadPrefixMask)];
        }

        Path childPath = inWorkItem.path;
        childPath.descend(threadInNodeGroup);

        NodeDesc childDesc;
        if (childHandle == sentinel) {
            childDesc.state = NodeState::Empty;
        } else if (childLevel == inDag.leaf_level()) {
            childDesc.state = NodeState::Partial;
        } else { // child is an inner node.
            childDesc.state = inDag.get_node_is_fully_filled(inDag.get_node_ptr(childLevel, childHandle)[0], childDesc.fillMaterial) ? NodeState::Filled : NodeState::Partial;
        }

        uint32_t childIntermediateMask;
        if constexpr (std::is_same_v<InEditorState, void>) {
            NodeEditDesc childEditDesc = editor.should_edit(childPath, childDepth, childDesc);

#if !EDITS_ENABLE_MATERIALS
            childEditDesc.fillMaterial = 0;
#endif

#if !FILL_FULL_REGION
            if (childEditDesc.operation == EditOperation::Fill || childEditDesc.operation == EditOperation::Empty)
                childEditDesc.operation = EditOperation::Subdivide;
#endif

            const bool subdivide = childEditDesc.operation == EditOperation::Subdivide;
            uint32_t outIndex;
            {
                // Only a single atomicAdd per warp (instead of per thread).
                const uint32_t subdivideMask = warp.ballot(subdivide);
                if (threadInWarp == 0)
                    outIndex = atomicAdd(pIndex, Utils::popc(subdivideMask));
                outIndex = warp.shfl(outIndex, 0) + Utils::popc(subdivideMask & warpThreadPrefixMask);
            }

            // Mask storing for each child whether it points to the intermediate SVO (existing SVDAG otherwise).
            childIntermediateMask = nodeGroup.ballot(subdivide);
            if (subdivide) {
                outWorkItems[outIndex] = { childPath, childHandle };
                childHandle = outIndex;
            } else if (childEditDesc.operation == EditOperation::Empty) {
                childHandle = sentinel;
            } else if (childEditDesc.operation == EditOperation::Fill) {
                childHandle = inDag.fullyFilledNodes.read(childLevel, childEditDesc.fillMaterial);
            }
        } else {
            OutEditorState childEditorState;
            NodeEditDesc childEditDesc = editor.should_edit(childPath, childDepth, childDesc, inWorkItem.editorState, childEditorState);

#if !FILL_FULL_REGION
            if (childEditDesc.operation == EditOperation::Fill || childEditDesc.operation == EditOperation::Empty)
                childEditDesc.operation = EditOperation::Subdivide;
#endif

            const bool subdivide = childEditDesc.operation == EditOperation::Subdivide;
            uint32_t outIndex;
            {
                // Only a single atomicAdd per warp (instead of per thread).
                const uint32_t subdivideMask = warp.ballot(subdivide);
                if (threadInWarp == 0)
                    outIndex = atomicAdd(pIndex, Utils::popc(subdivideMask));
                outIndex = warp.shfl(outIndex, 0) + Utils::popc(subdivideMask & warpThreadPrefixMask);
            }

            // Mask storing for each child whether it points to the intermediate SVO (existing SVDAG otherwise).
            childIntermediateMask = nodeGroup.ballot(subdivide);
            if (subdivide) {
                if constexpr (std::is_same_v<OutEditorState, void>)
                    outWorkItems[outIndex] = { childPath, childHandle };
                else
                    outWorkItems[outIndex] = { childPath, childHandle, childEditorState };
                childHandle = outIndex;
            } else if (childEditDesc.operation == EditOperation::Empty) {
                childHandle = sentinel;
            } else if (childEditDesc.operation == EditOperation::Fill) {
                childHandle = inDag.fullyFilledNodes.read(childLevel, childEditDesc.fillMaterial);
            }
        } // EditorState

        // Clear the node first such that after writing the variable sized node any unused bits are set to 0.
        // This is required for the SVDAG -> SVO conversion as it only considers IntermediateNode's as
        // duplicates if they are exactly the same, including the unused bits.
        for (uint32_t i = threadInNodeGroup; i < IntermediateSVO::Node::MaxSizeInU32; i += nodeGroup.size())
            outNode.padding[i] = 0;
        nodeGroup.sync(); // Make sure clear happens before write.

        const bool outputChild = childHandle != sentinel;
        const uint32_t childMask = nodeGroup.ballot(outputChild);
        if (childMask) {
            uint32_t* pChildren = DAG::encode_node_header_with_payload(&outNode.padding[0], childMask, childIntermediateMask);
            if (outputChild)
                pChildren[Utils::popc(childMask & nodeGroupThreadPrefixMask)] = childHandle;
            outNode.size = DAG::get_header_size() + Utils::popc(childMask);
        } else {
            outNode.size = 0;
        }
    } // Loop over inWorkItems
}

// clang-format off
template <typename T>
concept has_get_new_value_float3 = requires(T t) {
    { t.get_new_value(std::declval<float3>(), std::declval<uint32_t>()) } -> std::same_as<std::optional<uint32_t>>;
};
template <typename T>
concept has_get_new_value_uint3 = requires(T t) {
    { t.get_new_value(std::declval<uint3>(), std::declval<uint32_t>()) } -> std::same_as<std::optional<uint32_t>>;
};
template <typename T>
concept has_get_new_value_uint3_state = requires(T t, typename T::State state) {
    { t.get_new_value(std::declval<uint3>(), std::declval<uint32_t>(), state) } -> std::same_as<std::optional<uint32_t>>;
};
// clang-format on

template <typename DAG, typename Editor, typename EditorState>
static __global__ void createIntermediateSvoStructure_leaf_workgroup(
    const DAG inDag, const Editor editor,
    gsl::span<const CreateIntermediateWorkItem<EditorState>> inWorkItems, gsl::span<typename IntermediateSVO::Leaf> outLeaves)
{
    const auto workGroup = cooperative_groups::this_thread_block();
    const auto warp = cooperative_groups::tiled_partition<32>(workGroup);
    const uint32_t warpThreadRank = warp.thread_rank();
    const uint32_t warpInBlock = warp.meta_group_rank();

    __shared__ uint32_t reduceBuffer[2];
    __shared__ uint32_t leafBuffer[DAG::maxLeafSizeInU32];
    for (uint32_t workGroupIdx = blockIdx.x; workGroupIdx < inWorkItems.size(); workGroupIdx += gridDim.x) {
        const auto& inWorkItem = inWorkItems[workGroupIdx];
        auto& outLeaf = outLeaves[workGroupIdx];

        std::optional<uint32_t> optOldMaterial;
        if (inWorkItem.dagNodeHandle != sentinel) {
            const auto pInLeaf = inDag.get_leaf_ptr(inWorkItem.dagNodeHandle);
            uint32_t oldMaterial;
            if (inDag.get_material(pInLeaf, threadIdx.x, oldMaterial))
                optOldMaterial = oldMaterial;
        }

        Path voxelPath = inWorkItem.path;
        voxelPath.descend(threadIdx.x >> 3);
        voxelPath.descend(threadIdx.x & 0b111);
        const float3 position = voxelPath.as_position(0);

        NodeDesc nodeDesc;
        if (optOldMaterial.has_value()) {
            nodeDesc.state = NodeState::Filled;
            nodeDesc.fillMaterial = optOldMaterial.value();
        } else {
            nodeDesc.state = NodeState::Empty;
            nodeDesc.fillMaterial = 0xFFFFFFFF;
        }

        //const NodeEditDesc editDesc = editor.should_edit_impl(position, position + 1, nodeDesc);
        const NodeEditDesc editDesc = editor.should_edit(voxelPath, 0, nodeDesc);
        std::optional<uint32_t> optNewMaterial;
        if (editDesc.operation == EditOperation::Fill || editDesc.operation == EditOperation::Subdivide) {
            if constexpr (!std::is_void_v<EditorState> && has_get_new_value_uint3_state<Editor>)
                optNewMaterial = editor.get_new_value(voxelPath.path, optOldMaterial, inWorkItem.editorState);
            else if constexpr (has_get_new_value_uint3<Editor>)
                optNewMaterial = editor.get_new_value(voxelPath.path, optOldMaterial);
            else if constexpr (has_get_new_value_float3<Editor>)
                optNewMaterial = editor.get_new_value(position, optOldMaterial);
            else
                checkAlways(false);
        } else if (editDesc.operation == EditOperation::Keep) {
            optNewMaterial = optOldMaterial;
        }

        // For the DAG merging to work efficiently each item is assumed to be the same size.
        // Thus items smaller than the maximum size need to be padded consistently in order to correctly identify duplicates.
        if (threadIdx.x < DAG::maxLeafSizeInU32)
            leafBuffer[threadIdx.x] = 0;

        // Compute where to write the material.
        const bool store = optNewMaterial.has_value();
        // const uint32_t outputBit = cooperative_groups::reduce(workGroup, store ? DAG::MaterialBits : 0, thrust::plus<uint32_t>());
        // const uint32_t outputBit = reduce_sum(workGroup, reduceBuffer, store ? DAG::MaterialBits : 0);
        // Prefix sum
        const uint32_t storeMaskWarp = warp.ballot(store);
        const uint32_t prefixMaskWarp = (1u << warp.thread_rank()) - 1u;
        const uint32_t prefixSumWarp = Utils::popc(storeMaskWarp & prefixMaskWarp);
        reduceBuffer[warp.meta_group_rank()] = Utils::popc(storeMaskWarp);
        workGroup.sync();

        uint32_t prefixSumWorkGroup = prefixSumWarp;
        if (warp.meta_group_rank() == 1)
            prefixSumWorkGroup += reduceBuffer[0];
        const uint32_t voxelOffsetInBits = prefixSumWorkGroup * DAG::MaterialBits;
        const auto voxelOffsetInU32 = (voxelOffsetInBits >> 5);
        const auto voxelOffsetWithinU32 = (voxelOffsetInBits & 31);

        // Write header.
        leafBuffer[warp.meta_group_rank()] = warp.ballot(store);
        // Write materials.
        if constexpr (DAG::MaterialBits > 0) {
            if (store) {
                uint32_t newMaterial = optNewMaterial.value();
                atomicOr_block(&leafBuffer[2 + voxelOffsetInU32], newMaterial << voxelOffsetWithinU32);
                if constexpr (32 % DAG::MaterialBits != 0) {
                    if (voxelOffsetWithinU32 + DAG::MaterialBits > 32) {
                        newMaterial >>= 32 - voxelOffsetWithinU32;
                        atomicOr_block(&leafBuffer[3 + voxelOffsetInU32], newMaterial);
                    }
                }
            }
        }

        const auto numVoxels = reduceBuffer[0] + reduceBuffer[1];
        const auto leafSizeInU32 = 2 + (((numVoxels * DAG::MaterialBits) + 31) >> 5);
        outLeaf.size = leafSizeInU32;

        // Ensure all threads have written to leafBuffer and are done writing to reduceBuffer.
        workGroup.sync();
        __threadfence_block();

        static_assert(outLeaf.MaxSizeInU32 == DAG::maxLeafSizeInU32);
        if (threadIdx.x < outLeaf.MaxSizeInU32)
            outLeaf.padding[threadIdx.x] = leafBuffer[threadIdx.x];
        workGroup.sync(); // Ensure we read from leafBuffer before any thread continues into the next loop iteration.
    }
}

template <typename DAG, typename Editor, typename EditorState>
static __global__ void createIntermediateSvoStructure_leaf_cuda(
    const DAG inDag, const Editor editor,
    gsl::span<const CreateIntermediateWorkItem<EditorState>> inWorkItems, gsl::span<typename IntermediateSVO::Leaf> outLeaves)
{
    const auto globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalThreadIdx >= inWorkItems.size())
        return;

    const auto inWorkItem = inWorkItems[globalThreadIdx];

    const bool hasInLeaf = inWorkItem.dagNodeHandle != sentinel;
    const auto pInLeaf = hasInLeaf ? inDag.get_leaf_ptr(inWorkItem.dagNodeHandle) : typename DAG::LeafDecoder();
    typename DAG::LeafBuilder leafBuilder { outLeaves[globalThreadIdx].padding };

    // Iterate level 1 children
    for (uint8 child1 = 0; child1 < 8; child1++) {
        Path newPath1 = inWorkItem.path;
        newPath1.descend(child1);
        // Iterate level 2 children
        for (uint8 child2 = 0; child2 < 8; child2++) {
            Path newPath2 = newPath1;
            newPath2.descend(child2);
            const float3 position = newPath2.as_position(0);
            const uint32 voxelIndex = child1 * 8u + child2;

            uint32_t oldMaterial;
            NodeDesc nodeDesc;
            nodeDesc.state = (hasInLeaf && inDag.get_material(pInLeaf, voxelIndex, oldMaterial)) ? NodeState::Filled : NodeState::Empty;

            check(nodeDesc.state == NodeState::Empty || oldMaterial < 16);

            const NodeEditDesc editDesc = editor.should_edit_impl(position, position + 1, nodeDesc);
            if (editDesc.operation == EditOperation::Keep && nodeDesc.state == NodeState::Filled) {
                leafBuilder.set(oldMaterial);
            } else if (editDesc.operation == EditOperation::Fill || editDesc.operation == EditOperation::Subdivide) {
                std::optional<uint32_t> newMaterial;
                if constexpr (!std::is_void_v<EditorState> && has_get_new_value_uint3_state<Editor>)
                    newMaterial = editor.get_new_value(newPath2.path, oldMaterial, inWorkItem.editorState);
                else if constexpr (has_get_new_value_uint3<Editor>)
                    newMaterial = editor.get_new_value(newPath2.path, oldMaterial);
                else if constexpr (has_get_new_value_float3<Editor>)
                    newMaterial = editor.get_new_value(position, oldMaterial);
                else
                    checkAlways(false);

                if (newMaterial)
                    leafBuilder.set(*newMaterial);
            }

            leafBuilder.next();
        }
    }
    const uint32_t leafSizeInU32 = leafBuilder.finalize();
    for (uint32_t i = leafSizeInU32; i < IntermediateSVO::Leaf::MaxSizeInU32; ++i)
        outLeaves[globalThreadIdx].padding[i] = 0;
    outLeaves[globalThreadIdx].size = leafSizeInU32;

#if ENABLE_CHECKS
    const uint32_t* pOutLeaf = &outLeaves[globalThreadIdx].padding[0];
    if (pOutLeaf[0] == 0xFFFFFFFF && pOutLeaf[1] == 0xFFFFFFFF) {
        const float3 position = inWorkItem.path.as_position(0);
        const NodeDesc nodeDesc { .state = NodeState::Empty };
        const auto editDesc = editor.should_edit_impl(position, position + 4, nodeDesc);
        check(editDesc.operation != EditOperation::Fill);
    }
    check(outLeaves[globalThreadIdx].size <= Utils::sizeof_u32<typename IntermediateSVO::Leaf>());
#endif
}

template <typename DAG, typename Editor, typename EditorState>
void createIntermediateSvoStructure_leaf_recurse(
    const DAG& inDag, const Editor& editor, GpuMemoryPool& memPool, cudaStream_t stream, GPUTimingsManager& timingsManager,
    gsl::span<CreateIntermediateWorkItem<EditorState>> inWorkItems, IntermediateSVO& out)
{
    PROFILE_SCOPE("Traverse leaves");
    [[maybe_unused]] auto timing = timingsManager.timeScope("createIntermediateSvoStructure_leaf_cuda", stream);
    if (inWorkItems.size() > 0) {
        out.leaves = memPool.mallocAsync<typename IntermediateSVO::Leaf>(inWorkItems.size());
        // createIntermediateSvoStructure_leaf_cuda<DAG, Editor, EditorState><<<computeNumWorkGroups(inWorkItems.size()), workGroupSize, 0, stream>>>(
        //     inDag, editor, inWorkItems, out.leaves);
        static constexpr uint32_t maxWorkGroups = 8192 * 4;
        const uint32_t numWorkGroups = std::min((uint32_t)inWorkItems.size(), maxWorkGroups);
        createIntermediateSvoStructure_leaf_workgroup<DAG, Editor, EditorState><<<numWorkGroups, 64, 0, stream>>>(
            inDag, editor, inWorkItems, out.leaves);
    }
    memPool.freeAsync(inWorkItems);
    CUDA_CHECK_ERROR();
}

// clang-format off
template <typename T>
constexpr bool is_stateful_editor = requires(const T& editor) {
    typename T::State;
    { editor.createInitialState() } -> std::same_as<typename T::State>;
};
// clang-format on

template <bool, typename T>
struct conditional_edit_state {
    using type = void;
};
template <typename T>
struct conditional_edit_state<true, T> {
    using type = typename T::State;
};
template <typename T>
using conditional_edit_state_t = typename conditional_edit_state<is_stateful_editor<T>, T>::type;

template <typename Editor>
static constexpr bool useEditorState(uint32_t level)
{
    if constexpr (is_stateful_editor<Editor>) {
        const auto depth = MAX_LEVELS - level;
        return depth >= Editor::minStateFullDepth;
    } else {
        return false;
    }
};

template <typename DAG, typename Editor, typename EditorState, uint32_t level>
void createIntermediateSvoStructure_innerNode_recurse(
    const DAG& inDag, const Editor& editor, GpuMemoryPool& memPool, cudaStream_t stream, GPUTimingsManager& timingsManager,
    gsl::span<CreateIntermediateWorkItem<EditorState>> inWorkItems, IntermediateSVO& out)
{
    using OutEditorState = typename conditional_edit_state<useEditorState<Editor>(level + 1), Editor>::type;
    gsl::span<CreateIntermediateWorkItem<OutEditorState>> outWorkItems;
    {
        [[maybe_unused]] auto timing = timingsManager.timeScope("createIntermediateSvoStructure_innerNode_cuda", stream);
        using IntermediateNode = typename IntermediateSVO::Node;
        out.innerNodes[level] = memPool.mallocAsync<IntermediateNode>(inWorkItems.size());

        if (inWorkItems.size() > 0) {
            outWorkItems = memPool.mallocAsync<CreateIntermediateWorkItem<OutEditorState>>(8 * inWorkItems.size());
            printf("Size of work items: %u MB\n", uint32_t(outWorkItems.size_bytes() >> 20));

            uint32_t* pOutIndex = memPool.mallocAsync<uint32_t>();
            cudaMemsetAsync(pOutIndex, 0, sizeof(uint32_t), stream);
            CUDA_CHECK_ERROR();
            {
#if 0
                cudaMemsetAsync(out.innerNodes[level].data(), 0, out.innerNodes[level].size_bytes());
                createIntermediateSvoStructure_innerNode_cuda<DAG, Editor, EditorState, OutEditorState><<<computeNumWorkGroups(inWorkItems.size()), workGroupSize, 0, stream>>>(
                    inDag, editor, level, inWorkItems, out.innerNodes[level], outWorkItems, pOutIndex);
#else
                static constexpr uint32_t maxWorkGroups = 8192 * 4;
                static constexpr uint32_t warpsPerWorkGroup = 2;
                static constexpr uint32_t nodeGroupsPerWorkGroup = (32 * warpsPerWorkGroup) / 8;
                const uint32_t numWorkGroups = std::min((uint32_t)inWorkItems.size() / nodeGroupsPerWorkGroup + 1u, maxWorkGroups);
                createIntermediateSvoStructure_innerNode_warp<DAG, Editor, EditorState, OutEditorState><<<numWorkGroups, warpsPerWorkGroup * 32, 0, stream>>>(
                    inDag, editor, level, inWorkItems, out.innerNodes[level], outWorkItems, pOutIndex);
                CUDA_CHECK_ERROR();
#endif
            }
            memPool.freeAsync(inWorkItems);
            CUDA_CHECK_ERROR();

            // Read the number of child nodes that were output by the traversal step.
            uint32_t numOutputs;
            cudaMemcpy(&numOutputs, pOutIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            memPool.freeAsync(pOutIndex);
            outWorkItems = outWorkItems.subspan(0, numOutputs);
            CUDA_CHECK_ERROR();
        } else {
            memPool.freeAsync(inWorkItems);
        }

        if (outWorkItems.size() == 0) {
            memPool.freeAsync(outWorkItems);
            return;
        }
    }
    if constexpr (level + 1 == DAG::leaf_level()) {
        createIntermediateSvoStructure_leaf_recurse<DAG, Editor, OutEditorState>(
            inDag, editor, memPool, stream, timingsManager, outWorkItems, out);
    } else {
        createIntermediateSvoStructure_innerNode_recurse<DAG, Editor, OutEditorState, level + 1>(
            inDag, editor, memPool, stream, timingsManager, outWorkItems, out);
    }
}

template <typename DAG, typename Editor>
static IntermediateSVO createIntermediateSvoStructure_cuda(const DAG& inDag, const Editor& editor, GpuMemoryPool memPool, cudaStream_t stream, GPUTimingsManager& timingsManager)
{
    using EditorState = conditional_edit_state_t<Editor>;

    PROFILE_FUNCTION();

    CreateIntermediateWorkItem<EditorState> initialWorkItem;
    initialWorkItem.path = Path(0, 0, 0);
    initialWorkItem.dagNodeHandle = inDag.firstNodeIndex; // May be sentinel if there was no existing node covering this region.
    if constexpr (!std::is_same_v<EditorState, void>)
        initialWorkItem.editorState = editor.createInitialState();
    timingsManager.startTiming("createIntermediateSvoStructure_cuda setup", stream);
    auto workItems = memPool.mallocAsync<CreateIntermediateWorkItem<EditorState>>(1);
    cudaMemcpyAsync(workItems.data(), &initialWorkItem, sizeof(initialWorkItem), cudaMemcpyHostToDevice);
    timingsManager.endTiming("createIntermediateSvoStructure_cuda setup", stream);

    IntermediateSVO out {};
    createIntermediateSvoStructure_innerNode_recurse<DAG, Editor, EditorState, 0>(
        inDag, editor, memPool, stream, timingsManager, workItems, out);
    return out;
}

template <typename DAG, typename Editor>
IntermediateSVO create_edit_intermediate_svo(
    const DAG& inDag, const Editor& editor,
    GpuMemoryPool& memPool, cudaStream_t stream, GPUTimingsManager& timingsManager)
{
    PROFILE_FUNCTION();

    Stats stats;
    stats.start_work("Create intermediate edit tree");
    auto intermediateSVO = createIntermediateSvoStructure_cuda(inDag, editor, memPool, stream, timingsManager);
    stats.flush();
    deviceSynchronizeTiming();

    return intermediateSVO;
}

template IntermediateSVO create_edit_intermediate_svo(const MyGPUHashDAG<EMemoryType::GPU_Malloc>&, const MyGpuBoxEditor<true>&, GpuMemoryPool&, cudaStream_t, GPUTimingsManager&);
template IntermediateSVO create_edit_intermediate_svo(const MyGPUHashDAG<EMemoryType::GPU_Malloc>&, const MyGpuBoxEditor<false>&, GpuMemoryPool&, cudaStream_t, GPUTimingsManager&);
template IntermediateSVO create_edit_intermediate_svo(const MyGPUHashDAG<EMemoryType::GPU_Malloc>&, const MyGpuSphereEditor<true>&, GpuMemoryPool&, cudaStream_t, GPUTimingsManager&);
template IntermediateSVO create_edit_intermediate_svo(const MyGPUHashDAG<EMemoryType::GPU_Malloc>&, const MyGpuSphereEditor<false>&, GpuMemoryPool&, cudaStream_t, GPUTimingsManager&);
template IntermediateSVO create_edit_intermediate_svo(const MyGPUHashDAG<EMemoryType::GPU_Malloc>&, const MyGpuSpherePaintEditor&, GpuMemoryPool&, cudaStream_t, GPUTimingsManager&);
template IntermediateSVO create_edit_intermediate_svo(const MyGPUHashDAG<EMemoryType::GPU_Malloc>&, const MyGpuSphereErosionEditor<MyGPUHashDAG<EMemoryType::GPU_Malloc>>&, GpuMemoryPool&, cudaStream_t, GPUTimingsManager&);
template IntermediateSVO create_edit_intermediate_svo(const MyGPUHashDAG<EMemoryType::GPU_Malloc>&, const MyGpuCopyEditor<MyGPUHashDAG<EMemoryType::GPU_Malloc>>&, GpuMemoryPool&, cudaStream_t, GPUTimingsManager&);
