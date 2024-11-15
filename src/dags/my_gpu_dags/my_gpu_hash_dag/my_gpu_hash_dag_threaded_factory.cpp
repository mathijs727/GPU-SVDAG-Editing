#include "color_utils.h"
#include "configuration/gpu_hash_dag_definitions.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag_utils.h"
#include "dags/hash_dag/hash_table.h" // hash functions
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_factory.h"
#include "stats.h"
#include "typedefs.h"
#include "utils.h"
#include "voxel_textures.h"
#include <algorithm> // std::sort / std::copy
#include <array>
#include <deque>
#include <gsl/gsl-lite.hpp>
#include <limits>
#include <optional>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tuple>
#include <unordered_map>
#include <vector>

#if EDITS_ENABLE_MATERIALS
#define IMPORT_COLORS 1
static constexpr uint32_t fillMaterial = 5;
#else
#define IMPORT_COLORS 0
static constexpr uint32_t fillMaterial = 0;
#endif

namespace std {
template <typename T, typename S>
struct hash<std::pair<T, S>> {
    size_t operator()(const std::pair<T, S>& v) const
    {
        return std::hash<T>()(std::get<0>(v)) ^ std::hash<S>()(std::get<1>(v));
    }
};
}

[[maybe_unused]] static size_t traverseDebug(const BasicDAG& basicDag, const MyGPUHashDAG<EMemoryType::CPU>& dag, uint32_t level, uint32_t basicHandle, uint32_t handle)
{
    if (level == dag.leaf_level()) {
        const auto leaf = dag.get_leaf(handle).to_64();
        const auto basicLeaf = basicDag.get_leaf(basicHandle).to_64();
        checkAlways(leaf == basicLeaf);

        const auto pLeaf = dag.get_leaf_ptr(handle);
        for (uint32_t i = 0; i < 64; ++i) {
            uint32_t material;
            const bool isFilled = dag.get_material(pLeaf, i, material);
            if (leaf & (1llu << i)) {
                checkAlways(isFilled);
            } else {
                checkAlways(!isFilled);
            }
        }
        return dag.get_node_size(pLeaf) * sizeof(uint32_t);
    }

    const auto pNode = dag.get_node_ptr(level, handle);
    const auto bitmask = Utils::child_mask(pNode[0]);
    size_t dagMemoryInBytes = 0;
    for (uint8_t childIdx = 0; childIdx < 8; ++childIdx) {
        if (bitmask & (1 << childIdx)) {
            const auto basicChildIndex = basicDag.get_child_index(level, basicHandle, bitmask, childIdx);
            const auto childHandle = dag.get_child_index(level, handle, bitmask, childIdx);
            dagMemoryInBytes += traverseDebug(basicDag, dag, level + 1, basicChildIndex, childHandle);
        }
    }
    return dagMemoryInBytes + dag.get_node_size(pNode) * sizeof(uint32_t);
}

[[maybe_unused]] static float colorDistance(const float3& a, const float3& b)
{
    return length_squared(a - b);
}

template <typename Colors>
static std::vector<uint8_t> convertColorsToMaterials(const Colors& inColors, uint64_t firstColor, uint64_t numColors, const VoxelTextures& voxelTextures)
{
    PROFILE_FUNCTION();
    std::vector<uint8_t> out((size_t)numColors);
#if EDITS_ENABLE_MATERIALS
    const auto& colorLeaf = inColors.get_default_leaf();
    tbb::parallel_for(
        tbb::blocked_range<uint64_t>(0, numColors),
        [&](tbb::blocked_range<uint64_t> localRange) {
            for (uint64_t i = std::begin(localRange); i < std::end(localRange); ++i) {
                const float3 colorRGB = colorLeaf.get_color(firstColor + i).get_color();
                out[i] = (uint8_t)voxelTextures.getClosestMaterial(colorRGB);
            }
        });
#endif
    return out;
}

struct HashableElement {
    std::array<uint32_t, MyGPUHashDAG<EMemoryType::CPU>::maxItemSizeInU32> data;
    uint32_t length;

    bool operator==(const HashableElement& other) const
    {
        if (length != other.length)
            return false;
        return memcmp(data.data(), other.data.data(), length * sizeof(uint32_t)) == 0;
    }
};
struct HashableElementRef {
    HashableElement* pRef;

    bool operator==(const HashableElementRef& other) const
    {
        return *pRef == *other.pRef;
    }
};
struct ElementHasher {
    size_t operator()(const HashableElement& e) const
    {
        return Utils::murmurhash32xN(e.data.data(), e.length);
    }
};
struct ElementRefHasher {
    size_t operator()(const HashableElementRef& e) const
    {
        return hash(e);
    }
    static size_t hash(const HashableElementRef& e)
    {
        return Utils::murmurhash32xN(e.pRef->data.data(), e.pRef->length);
    }
    static bool equal(const HashableElementRef& lhs, const HashableElementRef& rhs)
    {
        return lhs == rhs;
    }
};

struct ConstructSubTree {
    uint32_t handle;
    std::optional<uint32_t> optFillMaterial;
};

static constexpr uint32_t parallelLevel = 4;

using ConstructElements = tbb::concurrent_vector<HashableElement>;
using ConstructDagMap = tbb::concurrent_hash_map<HashableElementRef, ConstructSubTree, ElementRefHasher>;

template <typename Colors>
static ConstructSubTree create_dag_cpu_threaded(
    const BasicDAG& sdag,
    const Colors& sdagColors,
    const VoxelTextures& voxelTextures,
    gsl::span<const uint8_t> materials,
    uint64_t& materialIndex,
    const uint32_t level,
    const uint32_t index,
    ConstructElements& outElements,
    ConstructDagMap& outElementsLUT)
{
    using TDAG = MyGPUHashDAG<EMemoryType::CPU>;

    std::optional<uint32_t> optFullyFilledMaterial;
    HashableElement element;
    if (!sdag.is_leaf(level)) {
        const uint32 node = sdag.get_node(level, index);
        const uint8 childMask = Utils::child_mask(node);
        check(Utils::popc(childMask) >= 1);

        uint32_t* pNode = element.data.data();
        pNode = TDAG::encode_node_header(pNode, childMask);

        const uint32_t childLevel = level + 1;
        std::array<uint32_t, 8> fullyFilledMaterials;
        std::fill(std::begin(fullyFilledMaterials), std::end(fullyFilledMaterials), SentinelMaterial);
        if (level < parallelLevel) {
            std::array<ConstructSubTree, 8> childSubTrees;
            tbb::task_group tg {};
            for (uint8_t i = 0; i < 8; ++i) {
                if (childMask & (1u << i)) {
                    const uint32 inChildIndex = sdag.get_child_index(level, index, childMask, i);
                    const uint64_t leavesInSubTree = sdagColors.get_leaves_count(childLevel, sdag.get_node(childLevel, inChildIndex));
                    tg.run([&, inChildIndex, leavesInSubTree, i, materialIndex]() {
                        if (level == parallelLevel - 1) {
                            const auto subTreeMaterials = convertColorsToMaterials(sdagColors, materialIndex, leavesInSubTree, voxelTextures);
                            uint64_t subTreeMaterialIndex = 0;
                            childSubTrees[i] = create_dag_cpu_threaded(sdag, sdagColors, voxelTextures, subTreeMaterials, subTreeMaterialIndex, childLevel, inChildIndex, outElements, outElementsLUT);
                        } else {
                            uint64_t subTreeMaterialIndex = materialIndex; // Don't want the child threads to modify materialIndex so make a copy.
                            childSubTrees[i] = create_dag_cpu_threaded(sdag, sdagColors, voxelTextures, materials, subTreeMaterialIndex, childLevel, inChildIndex, outElements, outElementsLUT);
                        }
                    });
                    materialIndex += leavesInSubTree;
                }
            }
            tg.wait();
            for (uint8_t i = 0; i < 8; ++i) {
                if (childMask & (1u << i)) {
                    const auto& childSubTree = childSubTrees[i];
                    *(pNode++) = childSubTree.handle;
                    if (childSubTree.optFillMaterial)
                        fullyFilledMaterials[i] = *childSubTree.optFillMaterial;
                }
            }
        } else {
            for (uint8 i = 0; i < 8; ++i) {
                if (childMask & (1u << i)) {
                    const uint32 inChildIndex = sdag.get_child_index(level, index, childMask, i);
                    const auto childSubTree = create_dag_cpu_threaded(sdag, sdagColors, voxelTextures, materials, materialIndex, childLevel, inChildIndex, outElements, outElementsLUT);
                    *(pNode++) = childSubTree.handle;
                    if (childSubTree.optFillMaterial)
                        fullyFilledMaterials[i] = *childSubTree.optFillMaterial;
                }
            }
        }
        element.length = (uint32_t)std::distance(element.data.data(), pNode);

        if (std::equal(std::begin(fullyFilledMaterials) + 1, std::end(fullyFilledMaterials), std::begin(fullyFilledMaterials)) && fullyFilledMaterials[0] != SentinelMaterial) {
            optFullyFilledMaterial = fullyFilledMaterials[0];
            TDAG::encode_node_header(element.data.data(), childMask, *optFullyFilledMaterial);
        }
    } else {
        const auto leaf = sdag.get_leaf(index);

        if constexpr (MyGPUHashDAG<EMemoryType::CPU>::NumMaterials == 0) {
            element.data[0] = leaf.low;
            element.data[1] = leaf.high;
            element.length = 2;
            if (leaf.low == 0xFFFFFFFF && leaf.high == 0xFFFFFFFF)
                optFullyFilledMaterial = 0;
        } else {
            typename TDAG::LeafBuilder leafBuilder { element.data.data() };

            bool allSameMaterial = true;
            uint32_t prevMaterial = SentinelMaterial;
            const uint64_t inLeafU64 = leaf.to_64();
            for (uint32_t i = 0; i < 64; ++i) {
                if (inLeafU64 & (1llu << i)) {
#if IMPORT_COLORS
                    const uint32_t material = materials[materialIndex++];
#else
                    const uint32_t material = fillMaterial;
#endif
                    leafBuilder.set(material);

                    if (i == 0)
                        prevMaterial = material;
                    allSameMaterial &= (material == prevMaterial);
                    prevMaterial = material;
                }
                leafBuilder.next();
            }
            element.length = leafBuilder.finalize();

            if (allSameMaterial && inLeafU64 == std::numeric_limits<uint64_t>::max())
                optFullyFilledMaterial = prevMaterial;
        }
    }

#if 0
    if (auto iter = outElementsLUT.find(HashableElementRef { &element }); iter != std::end(outElementsLUT)) {
#if ENABLE_CHECKS
        const auto& other = outElements[iter->second.handle].data;
        for (uint32_t i = 0; i < element.length; ++i) {
            check(element.data[i] == other[i]);
        }
#endif
        return iter->second;
    }
    ConstructSubTree subTree {
        .handle = (uint32_t)outElements.size(),
        .optFillMaterial = optFullyFilledMaterial
    };
    auto* pElement = &outElements.emplace_back(element);
    outElementsLUT.emplace(HashableElementRef { pElement }, subTree);
    return subTree;
#else
    {
        ConstructDagMap::const_accessor accessor {};
        if (outElementsLUT.find(accessor, HashableElementRef { &element }))
            return accessor->second;
    }

    auto elementIter = outElements.emplace_back(element);
    ConstructSubTree subTree {
        .handle = (uint32_t)std::distance(std::begin(outElements), elementIter),
        .optFillMaterial = optFullyFilledMaterial
    };
    // auto [iter, inserted] = outElementsLUT.try_emplace(HashableElementRef { &(*elementIter) }, subTree);
    ConstructDagMap::accessor accessor {};
    if (outElementsLUT.insert(accessor, HashableElementRef { &(*elementIter) }))
        accessor->second = subTree;
    return accessor->second;
#endif
}

static uint32_t convertToDagRecurse(
    const ConstructElements& inElements, uint32_t level, uint32_t index, std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t>& nodeCache, std::unordered_map<uint32_t, uint32_t>& leafCache, MyGPUHashDAG<EMemoryType::CPU>& out)
{
    HashableElement element = inElements[index];
    if (level == out.leaf_level()) {
        if (auto iter = leafCache.find(index); iter != std::end(leafCache)) {
            return iter->second;
        } else {
            // Prevent rare duplicate insertions.
            uint32_t leafHandle = out.find_leaf(element.data.data());
            if (leafHandle == out.invalid_handle)
                leafHandle = out.add_leaf(element.data.data());
            leafCache[index] = leafHandle;
            return leafHandle;
        }
    } else {
        const std::pair key { level, index };
        if (auto iter = nodeCache.find(key); iter != std::end(nodeCache)) {
            return iter->second;
        } else {
            const auto numChildren = Utils::popc(Utils::child_mask(element.data[0]));
            check(numChildren > 0 && numChildren <= 8);
            for (uint32_t i = 0; i < numChildren; ++i) {
                element.data[1 + i] = convertToDagRecurse(inElements, level + 1, element.data[1 + i], nodeCache, leafCache, out);
            }

            uint32_t nodeHandle = out.find_node(element.data.data());
            if (nodeHandle == out.invalid_handle)
                nodeHandle = out.add_node(element.data.data());
            nodeCache[key] = nodeHandle;
            return nodeHandle;
        }
    }
}

[[maybe_unused]] static MyGPUHashDAG<EMemoryType::CPU> convertToDag(const ConstructElements& elements, uint32_t rootNode, uint32_t targetLoadFactor)
{
    PROFILE_FUNCTION();
    using TDAG = MyGPUHashDAG<EMemoryType::CPU>;

    std::vector<uint32_t> hashTableSizes;
    {
        PROFILE_SCOPE("Determine hash table sizes");
        std::vector<uint32_t> elementsPerItemSize((size_t)TDAG::maxItemSizeInU32 + 1, 0);
        for (const auto& element : elements)
            elementsPerItemSize[element.length]++;

        for (uint32_t i = TDAG::minItemSizeInU32; i <= TDAG::maxItemSizeInU32; ++i) {
            hashTableSizes.push_back(std::max(1u, Utils::divideRoundUp(elementsPerItemSize[i], targetLoadFactor)));
        }
    }

    auto out = TDAG::allocate(hashTableSizes);
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> nodeCache;
    std::unordered_map<uint32_t, uint32_t> leafCache;
    out.firstNodeIndex = convertToDagRecurse(elements, 0, rootNode, nodeCache, leafCache, out);
    return out;
}

static ConstructSubTree create_dag_cpu(
    const BasicDAG& sdag,
    gsl::span<const uint8_t> materials,
    uint64_t& materialIndex,
    const uint32_t level,
    const uint32_t index,
    MyGPUHashDAG<EMemoryType::CPU>& out)
{
    using TDAG = MyGPUHashDAG<EMemoryType::CPU>;

    std::optional<uint32_t> optFullyFilledMaterial;
    HashableElement element;
    if (!sdag.is_leaf(level)) {
        const uint32 node = sdag.get_node(level, index);
        const uint8 childMask = Utils::child_mask(node);
        check(Utils::popc(childMask) >= 1);

        uint32_t* pNode = element.data.data();
        pNode = TDAG::encode_node_header(pNode, childMask);

        const uint32_t childLevel = level + 1;
        std::array<uint32_t, 8> fullyFilledMaterials;
        std::fill(std::begin(fullyFilledMaterials), std::end(fullyFilledMaterials), SentinelMaterial);
        for (uint8 i = 0; i < 8; ++i) {
            if (childMask & (1u << i)) {
                const uint32 inChildIndex = sdag.get_child_index(level, index, childMask, i);
                const auto childSubTree = create_dag_cpu(sdag, materials, materialIndex, childLevel, inChildIndex, out);
                *(pNode++) = childSubTree.handle;
                if (childSubTree.optFillMaterial)
                    fullyFilledMaterials[i] = *childSubTree.optFillMaterial;
            }
        }
        element.length = (uint32_t)std::distance(element.data.data(), pNode);

        if (std::equal(std::begin(fullyFilledMaterials) + 1, std::end(fullyFilledMaterials), std::begin(fullyFilledMaterials)) && fullyFilledMaterials[0] != SentinelMaterial) {
            optFullyFilledMaterial = fullyFilledMaterials[0];
            TDAG::encode_node_header(element.data.data(), childMask, *optFullyFilledMaterial);
        }

        uint32_t outHandle = out.find_node(element.data.data());
        if (outHandle == TDAG::invalid_handle)
            outHandle = out.add_node(element.data.data());
        return ConstructSubTree { .handle = outHandle, .optFillMaterial = optFullyFilledMaterial };
    } else {
        const auto leaf = sdag.get_leaf(index);

        if constexpr (MyGPUHashDAG<EMemoryType::CPU>::NumMaterials == 0) {
            element.data[0] = leaf.low;
            element.data[1] = leaf.high;
            element.length = 2;
            if (leaf.low == 0xFFFFFFFF && leaf.high == 0xFFFFFFFF)
                optFullyFilledMaterial = 0;
        } else {
            typename TDAG::LeafBuilder leafBuilder { element.data.data() };

            bool allSameMaterial = true;
            uint32_t prevMaterial = SentinelMaterial;
            const uint64_t inLeafU64 = leaf.to_64();
            for (uint32_t i = 0; i < 64; ++i) {
                if (inLeafU64 & (1llu << i)) {
#if IMPORT_COLORS
                    const uint32_t material = materials[materialIndex++];
#else
                    const uint32_t material = fillMaterial;
#endif
                    leafBuilder.set(material);

                    if (i == 0)
                        prevMaterial = material;
                    allSameMaterial &= (material == prevMaterial);
                    prevMaterial = material;
                }
                leafBuilder.next();
            }
            element.length = leafBuilder.finalize();

            if (allSameMaterial && inLeafU64 == std::numeric_limits<uint64_t>::max())
                optFullyFilledMaterial = prevMaterial;
        }

        uint32_t outHandle = out.find_leaf(element.data.data());
        if (outHandle == TDAG::invalid_handle)
            outHandle = out.add_leaf(element.data.data());
        return ConstructSubTree { .handle = outHandle, .optFillMaterial = outHandle };
    }
}

template <typename Colors>
static void load_from_DAG(MyGPUHashDAG<EMemoryType::GPU_Malloc>& outDag, const BasicDAG& inDag, const Colors& inColors, const VoxelTextures& inVoxelTextures)
{
    PROFILE_FUNCTION();
    SCOPED_STATS("Creating GPU hash dag");

    Stats stats;

#if 0
    stats.start_work("Convert colors to materials");
    std::vector<uint8_t> materials;
    {
        PROFILE_SCOPE("Convert colors to materials");
#if EDITS_ENABLE_MATERIALS
        materials = convertColorsToMaterials(inColors, 0, inColors.get_leaves_count(0, inDag.get_node(0, 0)), inVoxelTextures);
#endif
    }

    stats.start_work("Construct DAG");
    auto cpuDag = MyGPUHashDAG<EMemoryType::CPU>::allocate(1024 * 1024);
    {
        uint64_t materialIndex = 0;
        cpuDag.firstNodeIndex = create_dag_cpu(inDag, materials, materialIndex, 0, inDag.get_first_node_index(), cpuDag).handle;
    }
#else
    ConstructElements elements;
    ConstructDagMap elementsLUT;
    uint32_t tbbRootNode;
    stats.start_work("Construct DAG using Intel TBB");
    {
        PROFILE_SCOPE("Construct DAG using Intel TBB");
        uint64_t materialIndex = 0;
        std::vector<uint8_t> materials;
        tbbRootNode = create_dag_cpu_threaded(inDag, inColors, inVoxelTextures, materials, materialIndex, 0, inDag.get_first_node_index(), elements, elementsLUT).handle;
    }
    elementsLUT.clear();

    stats.start_work("Convert to final DAG");
    auto cpuDag = convertToDag(elements, tbbRootNode, TARGET_LOAD_FACTOR);
#endif

#if ENABLE_CHECKS
    traverseDebug(inDag, cpuDag, 0, inDag.get_first_node_index(), cpuDag.get_first_node_index());
#endif

    outDag = cpuDag.copy<EMemoryType::GPU_Malloc>();
    cpuDag.free();
}

void MyGPUHashDAGFactory::load_from_DAG(MyGPUHashDAG<EMemoryType::GPU_Malloc>& outDag, const BasicDAG& inDag, const BasicDAGCompressedColors& inColors, const VoxelTextures& inVoxelTextures)
{
    ::load_from_DAG(outDag, inDag, inColors, inVoxelTextures);
}

void MyGPUHashDAGFactory::load_from_DAG(MyGPUHashDAG<EMemoryType::GPU_Malloc>& outDag, const BasicDAG& inDag, const BasicDAGUncompressedColors& inColors, const VoxelTextures& inVoxelTextures)
{
    ::load_from_DAG(outDag, inDag, inColors, inVoxelTextures);
}
