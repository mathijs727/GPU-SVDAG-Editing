#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_factory.h"
#include "color_utils.h"
#include "configuration/gpu_hash_dag_definitions.h"
#include "configuration/profile_definitions.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag_utils.h"
#include "dags/hash_dag/hash_table.h" // hash functions
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "stats.h"
#include "typedefs.h"
#include "utils.h"
#include "voxel_textures.h"
#include <algorithm> // std::sort / std::copy
#include <array>
#include <cstring>
#include <cuda.h>
#include <fstream>
#include <gsl/gsl-lite.hpp>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if EDITS_ENABLE_MATERIALS
#define IMPORT_COLORS 1
static constexpr uint32_t fillMaterial = 5;
#else
#define IMPORT_COLORS 0
static constexpr uint32_t fillMaterial = 0;
#endif

struct TupleHasher {
    template <typename... Ts>
    size_t operator()(const std::tuple<Ts...>& s) const noexcept
    {
        return std::apply([](auto... v) { return (std::hash<Ts>()(v) ^ ...); }, s);
    }
};

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

struct ConstructionStats {
    size_t nodeCount = 0;
    size_t leafCount = 0;
    size_t uniqueNodeCount = 0;
    size_t uniqueLeafCount = 0;
    size_t nodeMemoryInBytes = 0;
    size_t leafMemoryInBytes = 0;
    size_t uniqueNodeMemoryInBytes = 0;
    size_t uniqueLeafMemoryInBytes = 0;

    template <EMemoryType memoryType>
    void print(const MyGPUHashDAG<memoryType>& dag) const
    {
        printf("\n=== CONSTRUCTION STATS ===\n");
        printf("DAG compression (#nodes):  %zu => %zu (%.2f%%)\n", nodeCount, uniqueNodeCount, (double)uniqueNodeCount / (double)nodeCount * 100.0);
        printf("DAG compression (#leaves): %zu => %zu (%.2f%%)\n", leafCount, uniqueLeafCount, (double)uniqueLeafCount / (double)leafCount * 100.0);

        printf("DAG compression (nodes bytes):  %zuMB => %zuMB (%.2f%%)\n", nodeMemoryInBytes >> 20, uniqueNodeMemoryInBytes >> 20, (double)uniqueNodeMemoryInBytes / (double)nodeMemoryInBytes * 100.0);
        printf("DAG compression (leaves bytes): %zuMB => %zuMB (%.2f%%)\n", leafMemoryInBytes >> 20, uniqueLeafMemoryInBytes >> 20, (double)uniqueLeafMemoryInBytes / (double)leafMemoryInBytes * 100.0);

        printf("Total allocated memory: %.2f MB\n", Utils::to_MB(dag.memory_allocated()));
        printf("Items:                  %.2f MB\n", Utils::to_MB(dag.memory_used_by_items()));
        printf("Tables and slabs:       %.2f MB\n", Utils::to_MB(dag.memory_used_by_slabs()));
        printf("Dag items memory = %zu B\n", dag.memory_used_by_items());
        printf("Expected memory  = %zu B\n", uniqueNodeMemoryInBytes + uniqueLeafMemoryInBytes);
    }

    template <EMemoryType memoryType>
    void write_json(const MyGPUHashDAG<memoryType>& dag, std::ostream& stream) const
    {
        nlohmann::json jsonStats;
        jsonStats["svo_node_count"] = nodeCount;
        jsonStats["svdag_node_count"] = uniqueNodeCount;
        jsonStats["svo_node_memory_in_bytes"] = nodeMemoryInBytes;
        jsonStats["svdag_node_memory_in_bytes"] = uniqueNodeMemoryInBytes;

        jsonStats["svo_leaf_count"] = leafCount;
        jsonStats["svdag_leaf_count"] = uniqueLeafCount;
        jsonStats["svo_leaf_memory_in_bytes"] = leafMemoryInBytes;
        jsonStats["svdag_leaf_memory_in_bytes"] = uniqueLeafMemoryInBytes;

        jsonStats["svdag_memory_allocated"] = dag.memory_allocated();
        jsonStats["svdag_memory_used_by_items"] = dag.memory_used_by_items();
        jsonStats["svdag_memory_used_by_slabs"] = dag.memory_used_by_slabs();

        nlohmann::json jsonOut;
        jsonOut["settings"] = getDefineInfoJson();
        jsonOut["machine"] = getSystemInfoJson();
        jsonOut["stats"] = jsonStats;

        stream << std::setfill(' ') << std::setw(4) << jsonOut;
    }
};

// There are "only" 16 million colors (2^24 for RGB8) thus this look-up table is relatively small (64MB).
std::vector<uint32_t> createColorToMaterialLookUpTable(const VoxelTextures& palette)
{
    static constexpr size_t numColors = 1u << 24;

    std::vector<uint32_t> materials(numColors);
    for (uint32_t i = 0; i < numColors; ++i) {
        const auto colorRGB = ColorUtils::rgb888_to_float3(i);
        materials[i] = palette.getClosestMaterial(colorRGB);
    }
    return materials;
}

#if EDITS_ENABLE_MATERIALS
template <typename Colors>
class MaterialDecoder {
public:
    MaterialDecoder(const BasicDAG& dag, const Colors& colors, const VoxelTextures& palette)
        : m_colorToMaterialLUT(createColorToMaterialLookUpTable(palette))
        , m_colors(colors.get_default_leaf())
        , m_globalIdx(0)
        , m_localIdx(0)
        , m_numColors(colors.get_leaves_count(0, dag.get_node(0, dag.get_first_node_index())))
    {
        m_cache.resize(CacheSize);
        updateCache();
    }

    uint32_t getNextMaterial()
    {
        // const float3 colorRGB = m_colors.get_color(m_colorIdx++).get_color();
        // const uint32_t colorRGB8 = ColorUtils::float3_to_rgb888(colorRGB);
        //  return m_colorToMaterialLUT[colorRGB8 & 0x00FFFFFF];

        if (m_localIdx >= m_cache.size())
            updateCache();
        return m_cache[m_localIdx++];
    }

private:
    void updateCache()
    {
        m_globalIdx += m_localIdx;
        m_localIdx = 0;

        const uint64_t numColorsToGo = std::min(m_numColors - m_globalIdx, CacheSize);
        tbb::parallel_for(
            tbb::blocked_range<uint64_t>(0, numColorsToGo),
            [&](tbb::blocked_range<uint64_t> localRange) {
                for (uint64_t i = std::begin(localRange); i != std::end(localRange); ++i) {
                    const float3 colorRGB = m_colors.get_color(m_globalIdx + i).get_color();
                    const uint32_t colorRGB8 = ColorUtils::float3_to_rgb888(colorRGB);
                    m_cache[i] = m_colorToMaterialLUT[colorRGB8 & 0x00FFFFFF];
                }
            });
    }

private:
    static constexpr uint64_t CacheSize = 128 * 1024 * 1024;
    std::vector<uint32_t> m_colorToMaterialLUT;
    std::vector<uint32_t> m_cache;

    Colors::ColorLeaf m_colors;
    uint64_t m_globalIdx, m_localIdx, m_numColors;
};
#else
template <typename Colors>
class MaterialDecoder {
public:
    MaterialDecoder(const BasicDAG& dag, const Colors& colors, const VoxelTextures& palette) { }
    uint32_t getNextMaterial() { return 0; }
};
#endif

class IntermediateSVDAG : public MyGPUHashDAG<EMemoryType::CPU> {
public:
    IntermediateSVDAG(const MyGPUHashDAG<EMemoryType::CPU>& parent)
        : MyGPUHashDAG(parent)
    {
    }

    uint32_t find_node(const uint32_t* pItem) const
    {
        const uint32_t out = find(pItem, get_node_size(pItem));
        // check(this->Super::find_node(pItem) == out);
        return out;
    }
    uint32_t find_leaf(const uint32_t* pItem) const
    {
        const uint32_t out = find(pItem, get_leaf_size(pItem));
        // check(this->Super::find_leaf(pItem) == out);
        return out;
    }

    uint32_t add_node(const uint32_t* pItem)
    {
        const uint32_t handle = this->Super::add_node(pItem);
        const HashableItemRef<NodeDecoder> key { this->Super::get_node_ptr(0, handle), get_node_size(pItem) };
        m_itemsCache[key] = handle;
        return handle;
    }
    uint32_t add_leaf(const uint32_t* pItem)
    {
        const uint32_t handle = this->Super::add_leaf(pItem);
        const HashableItemRef<NodeDecoder> key { this->Super::get_leaf_ptr(handle), get_leaf_size(pItem) };
        m_itemsCache[key] = handle;
        return handle;
    }

    size_t memory_allocated() const
    {
        size_t out = this->Super::memory_allocated();
        out += m_itemsCache.size() * sizeof(std::remove_all_extents_t<decltype(m_itemsCache)>::value_type);
        return out;
    }

private:
    uint32_t find(const uint32_t* pItem, uint32_t itemSizeInU32) const
    {
        HashableItemRef<const uint32_t*> hashItem { pItem, itemSizeInU32 };
        if (auto iter = m_itemsCache.find<HashableItemRef<const uint32_t*>>(hashItem); iter != std::end(m_itemsCache)) {
            return iter->second;
        } else {
            return invalid_handle;
        }
    }

private:
    template <typename T>
    struct HashableItemRef {
        T pointer;
        uint32_t sizeInU32;

        HashableItemRef(T pointer, uint32_t sizeInU32)
            : pointer(pointer)
            , sizeInU32(sizeInU32)
        {
        }

        template <typename S>
        bool operator==(const HashableItemRef<S>& other) const
        {
            if (sizeInU32 != other.sizeInU32)
                return false;
            for (uint32_t i = 0; i < sizeInU32; ++i) {
                if (pointer[i] != other.pointer[i])
                    return false;
            }
            return true;
        }
    };
    using Super = MyGPUHashDAG<EMemoryType::CPU>;
    struct Hasher {
        using is_transparent = void;

        template <typename S>
        size_t operator()(const HashableItemRef<S>& item) const
        {
            return Utils::murmurhash32xN(item.pointer, item.sizeInU32);
        }
    };
    static_assert(std::is_same_v<LeafDecoder, NodeDecoder>);
    std::unordered_map<HashableItemRef<NodeDecoder>, uint32_t, Hasher, std::equal_to<>> m_itemsCache;
};

struct ConstructSubTree {
    uint32_t handle;
    std::optional<uint32_t> optFillMaterial;
};

// using ConstructionDAG =  MyGPUHashDAG<EMemoryType::CPU>;
using ConstructionDAG = IntermediateSVDAG;

template <typename Colors>
struct ConstructArgs {
    const BasicDAG& sdag;
    MaterialDecoder<Colors>& sdagColors;
    ConstructionDAG& hashDag;
    ConstructionStats& stats;
};

template <typename Colors>
static ConstructSubTree create_dag_cpu(
    // const BasicDAG& sdag,
    // MaterialDecoder<Colors>& sdagColors,
    ConstructArgs<Colors>& args,
    const uint32_t level,
    const uint32_t index)
// ConstructionDAG& outHashDag,
// ConstructionStats& outStats)
{
    using TDAG = MyGPUHashDAG<EMemoryType::CPU>;
    std::optional<uint32_t> optFullyFilledMaterial;

    if (level == 4) {
        const auto memoryUsage = args.stats.uniqueNodeMemoryInBytes + args.stats.uniqueLeafMemoryInBytes;
        printf("Memory usage: %zu MB\n", memoryUsage >> 20);
        printf("Memory allocated: %zu MB\n", args.hashDag.memory_allocated() >> 20);
    }

    if (!args.sdag.is_leaf(level)) {
        const uint32 node = args.sdag.get_node(level, index);
        const uint8 childMask = Utils::child_mask(node);
        check(Utils::popc(childMask) >= 1);

        uint32_t nodeBuffer[10];
        uint32_t* pNode = nodeBuffer;
        pNode = args.hashDag.encode_node_header(pNode, childMask);

        std::array<uint32_t, 8> fullyFilledMaterials;
        std::fill(std::begin(fullyFilledMaterials), std::end(fullyFilledMaterials), SentinelMaterial);
        for (uint8 i = 0; i < 8; ++i) {
            if (childMask & (1u << i)) {
                const uint32 inChildIndex = args.sdag.get_child_index(level, index, childMask, i);
                const auto childSubTree = create_dag_cpu(args, level + 1, inChildIndex);
                *(pNode++) = childSubTree.handle;
                if (childSubTree.optFillMaterial)
                    fullyFilledMaterials[i] = *childSubTree.optFillMaterial;
            }
        }

        if (std::equal(std::begin(fullyFilledMaterials) + 1, std::end(fullyFilledMaterials), std::begin(fullyFilledMaterials)) && fullyFilledMaterials[0] != SentinelMaterial) {
            optFullyFilledMaterial = fullyFilledMaterials[0];
            TDAG::encode_node_header(nodeBuffer, childMask, *optFullyFilledMaterial);
        }

        check(Utils::child_mask(nodeBuffer[0]) != 0);
        const auto nodeSizeInBytes = args.hashDag.get_node_size(nodeBuffer) * sizeof(uint32_t);
        args.stats.nodeCount++;
        args.stats.nodeMemoryInBytes += nodeSizeInBytes;
        if (uint32_t nodeHandle = args.hashDag.find_node(nodeBuffer); nodeHandle != args.hashDag.invalid_handle) {
            return { .handle = nodeHandle, .optFillMaterial = optFullyFilledMaterial };
        } else {
            args.stats.uniqueNodeCount++;
            args.stats.uniqueNodeMemoryInBytes += nodeSizeInBytes;
            nodeHandle = args.hashDag.add_node(nodeBuffer);
            return { .handle = nodeHandle, .optFillMaterial = optFullyFilledMaterial };
        }
    } else {
        const auto leaf = args.sdag.get_leaf(index);

        uint32_t leafBuffer[args.hashDag.maxItemSizeInU32];
        if constexpr (MyGPUHashDAG<EMemoryType::CPU>::NumMaterials == 0) {
            leafBuffer[0] = leaf.low;
            leafBuffer[1] = leaf.high;
            if (leaf.low == 0xFFFFFFFF && leaf.high == 0xFFFFFFFF)
                optFullyFilledMaterial = 0;
        } else {
            typename TDAG::LeafBuilder leafBuilder { leafBuffer };

            bool allSameMaterial = true;
            uint32_t prevMaterial = SentinelMaterial;
            const uint64_t inLeafU64 = leaf.to_64();
            for (uint32_t i = 0; i < 64; ++i) {
                if (inLeafU64 & (1llu << i)) {
#if IMPORT_COLORS
                    const uint32_t material = args.sdagColors.getNextMaterial();
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
            leafBuilder.finalize();

            if (allSameMaterial && inLeafU64 == std::numeric_limits<uint64_t>::max())
                optFullyFilledMaterial = prevMaterial;
        }

        check(leafBuffer[0] != 0 || leafBuffer[1] != 0);
        const auto leafSizeInBytes = args.hashDag.get_leaf_size(leafBuffer) * sizeof(uint32_t);
        args.stats.leafCount++;
        args.stats.leafMemoryInBytes += leafSizeInBytes;
        if (uint32_t leafHandle = args.hashDag.find_leaf(leafBuffer); leafHandle != args.hashDag.invalid_handle) {
            return { .handle = leafHandle, .optFillMaterial = optFullyFilledMaterial };
        } else {
            args.stats.uniqueLeafCount++;
            args.stats.uniqueLeafMemoryInBytes += leafSizeInBytes;

            leafHandle = args.hashDag.add_leaf(leafBuffer);
            return { .handle = leafHandle, .optFillMaterial = optFullyFilledMaterial };
        }
    }
}

template <typename Colors>
static void load_from_DAG(MyGPUHashDAG<EMemoryType::GPU_Malloc>& outDag, const BasicDAG& inDag, const Colors& inColors, const VoxelTextures& inVoxelTextures)
{
    PROFILE_FUNCTION();
    SCOPED_STATS("Creating GPU hash dag");

    static constexpr size_t averageNodeSizeInU32 = 8;
    static constexpr size_t numNodeSizes = 9 - 2; // 1..8 children + bitmask
    static constexpr size_t initialTargetLoadFactor = 64; // Hash table is a lot more efficient on GPU than on CPU.
    const size_t numNodesEstimate = inDag.data.size() / averageNodeSizeInU32;
    const uint32_t initialTableSizes = (uint32_t)((numNodesEstimate / initialTargetLoadFactor) / numNodeSizes) + 1u;
    printf("initialTableSizes = %u\n", initialTableSizes);

    Stats stats;
    stats.start_work("Constructing GPUHashDAG in CPU memory");
    auto cpuDag = MyGPUHashDAG<EMemoryType::CPU>::allocate(initialTableSizes);
    cpuDag.checkHashTables();

    ConstructionStats constructionStats {};
    {
        PROFILE_SCOPE("construct GPUHashDAG CPU");
        constructionStats.nodeMemoryInBytes = constructionStats.uniqueNodeMemoryInBytes = cpuDag.memory_used_by_items(); // May contain some initial fully-filled nodes.
        printf("Initial cpuDag size: %zu MiB\n", cpuDag.memory_allocated() >> 20);
        MaterialDecoder<Colors> materialDecoder(inDag, inColors, inVoxelTextures);
        if constexpr (std::is_same_v<ConstructionDAG, IntermediateSVDAG>) {
            IntermediateSVDAG intermediateDAG = cpuDag;
            ConstructArgs args { inDag, materialDecoder, intermediateDAG, constructionStats };
            intermediateDAG.firstNodeIndex = create_dag_cpu(args, 0, inDag.get_first_node_index()).handle;
            cpuDag = intermediateDAG;
        } else {
            ConstructArgs args { inDag, materialDecoder, cpuDag, constructionStats };
            cpuDag.firstNodeIndex = create_dag_cpu(args, 0, inDag.get_first_node_index()).handle;
        }
#if ENABLE_CHECKS
        cpuDag.checkHashTables();
#endif
        printf("Initial DAG size: %zu MiB\n", cpuDag.memory_allocated() >> 20);
        constructionStats.print(cpuDag);
    }

#if CAPTURE_MEMORY_STATS_SLOW
    const auto dagSizeInBytes = cpuDag.memory_used_by_items();
    checkAlways(dagSizeInBytes == constructionStats.uniqueLeafMemoryInBytes + constructionStats.uniqueNodeMemoryInBytes);
#endif

    stats.start_work("Resizing GPUHashDAG to target load factor");
    {
        PROFILE_SCOPE("Resize GPUHashDAG");
        cpuDag.setLoadFactor(TARGET_LOAD_FACTOR);
        constructionStats.print(cpuDag);

        // May be slightly different due to nodes/leaves moving to a different location in memory.
        // Updating their pointers will update the parent's pointers which introduces the possibility
        // that they become a duplicate of a leaf (that happens to have the exact same memory pattern).
        // const auto newDagSizeInBytes = cpuDag.memory_used_by_items();
        // checkAlways(newDagSizeInBytes == dagSizeInBytes);
    }

#if defined(CONSTRUCT_STATS_FILE_PATH)
    {
        std::filesystem::path profilingPath { PROFILING_PATH };
        if (!std::filesystem::exists(profilingPath))
            std::filesystem::create_directories(profilingPath);
        std::ofstream file { profilingPath / std::filesystem::path(std::string(CONSTRUCT_STATS_FILE_PATH) + ".json") };
        constructionStats.write_json(cpuDag, file);
    }
#endif

#if ENABLE_CHECKS
    traverseDebug(inDag, cpuDag, 0, inDag.get_first_node_index(), cpuDag.get_first_node_index());
#endif

    outDag = cpuDag.copy<EMemoryType::GPU_Malloc>();

    /*{
        PROFILE_SCOPE("Garbage Collect GPUHashDAG");
        std::array rootNodes { outDag.get_first_node_index() };
        outDag.garbageCollect(rootNodes);
        constructionStats.print(outDag);
    }*/

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
