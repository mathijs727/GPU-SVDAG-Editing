//
#include <CLI/CLI.hpp>
#undef check
#define JSON_HAS_RANGES 0
#include <nlohmann/json.hpp>
// Include before typdefs (CLI also defines check())
#include "DyCuckoo/core/dynamic_cuckoo.cuh"
#include "DyCuckoo/core/static_cuckoo.cuh"
#include "DyCuckoo/data/data_layout.cuh"
#include "array.h"
#include "cuda_helpers.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/atomic64_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/compact_acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/individual_chaining_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/ticket_board_hash_table.h"
#include "my_units.h"
#include "stats.h"
#include "timings.h"
#include "utils.h"
#include <algorithm>
#include <chrono>
#include <concepts>
#include <cstdio>
#include <execution>
#include <filesystem>
#include <fstream>
#include <gpu_hash_table.cuh>
#include <numeric>
#include <optional>
#include <random>
#include <slab_hash.cuh>
#include <string>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <unordered_map>
#include <variant>
#include <vector>

#define CHECK_OUTPUT_FOR_CORRECTNESS 0

constexpr uint32_t not_found = 0xFFFFFFFF;
constexpr uint32_t threadPerWarp = 32;
enum class ThreadType {
    Thread,
    Warp,
    WarpHybrid
};
// For some reason magic_enum::enum_name() is broken and always returns nullptr.
// Manual implementation as a quick work-around.
std::string toString(ThreadType threadType)
{
    if (threadType == ThreadType::Thread)
        return "thread";
    else if (threadType == ThreadType::Warp)
        return "warp";
    else if (threadType == ThreadType::WarpHybrid)
        return "warp_hybrid";

    checkAlways(false);
    return "";
}
std::string toString(HashMethod hashMethod)
{
    if (hashMethod == HashMethod::Murmur)
        return "Murmur";
    if (hashMethod == HashMethod::MurmurXor)
        return "MurmurXor";
    else if (hashMethod == HashMethod::SlabHashXor)
        return "SlabHashXor";
    else if (hashMethod == HashMethod::SlabHashBoostCombine)
        return "SlabHashBoostCombine";
    else if (hashMethod == HashMethod::SlabHashSingle)
        return "SlabHashSingle";

    checkAlways(false);
    return "";
}
struct TestSettings {
    uint32_t itemSizeInU32;
    uint32_t numInitialItems;
    uint32_t numInsertItems;
    uint32_t numReservedItems;
    uint32_t numSearchItems;
    uint32_t numBuckets;

    uint32_t warpsPerWorkGroup;
    uint32_t maxNumWorkGroups;

    uint32_t numRuns;

    std::optional<std::filesystem::path> optOutFilePath;

    void print() const
    {
        printf("itemSizeInU32       = %u\n", itemSizeInU32);
        printf("numInitialItems     = %u\n", numInitialItems);
        printf("numInsertItems      = %u\n", numInsertItems);
        printf("numTotalItems       = %u\n", numInitialItems + numInsertItems);
        printf("numSearchItems      = %u\n", numSearchItems);
        printf("numBuckets          = %u\n", numBuckets);
        printf("numReservedItems    = %u\n", numReservedItems);
        printf("warpsPerWorkGroup   = %u\n", warpsPerWorkGroup);
        printf("maxNumWorkGroups    = %u\n", maxNumWorkGroups);
        printf("numRuns             = %u\n", numRuns);
    }

    nlohmann::json toJson() const
    {
        nlohmann::json out {};
        out["itemSizeInU32"] = itemSizeInU32;
        out["numInitialItems"] = numInitialItems;
        out["numInsertItems"] = numInsertItems;
        out["numSearchItems"] = numSearchItems;
        out["numBuckets"] = numBuckets;
        out["numReservedItems"] = numReservedItems;
        out["warpsPerWorkGroup"] = warpsPerWorkGroup;
        out["maxNumWorkGroups"] = maxNumWorkGroups;
        out["numRuns"] = numRuns;
        return out;
    }
};

template <typename Table>
static __global__ void add_kernel(Table table, gsl::span<const uint32_t> inItems, uint32_t numItems, uint32_t itemSizeInU32)
{
    const uint32_t gridSize = blockDim.x * gridDim.x;
    for (uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; globalThreadIdx < numItems; globalThreadIdx += gridSize) {
        const uint32_t* pItem = &inItems[globalThreadIdx * itemSizeInU32];
        table.add(pItem);
    }
}
template <typename Table>
static __global__ void addAsWarpHybrid_kernel(Table hashTable, gsl::span<const uint32_t> inItems, uint32_t numItems, uint32_t itemSizeInU32)
{
    const uint32_t gridSize = blockDim.x * gridDim.x;
    const uint32_t end = ((uint32_t)inItems.size() + 31) & (~31);
    for (uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; globalThreadIdx < end; globalThreadIdx += gridSize) {
        check(__activemask() == 0xFFFFFFFF);
        const bool valid = globalThreadIdx < numItems;
        const uint32_t* pItem = valid ? &inItems[globalThreadIdx * itemSizeInU32] : nullptr;
        hashTable.addAsWarpHybrid(pItem, valid);
    }
}
template <typename Table>
static __global__ void addAsWarp_kernel(Table hashTable, gsl::span<const uint32_t> inItems, uint32_t numItems, uint32_t itemSizeInU32)
{
#ifdef __CUDA_ARCH__
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < numItems + warpInBlock; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx < numItems) {
            const uint32_t* pItem = &inItems[warpIdx * itemSizeInU32];
            hashTable.addAsWarp(pItem);
        }
    }
#endif
}

template <typename Table>
static __global__ void find_kernel(Table hashTable, gsl::span<const uint32_t> inItems, uint32_t numItems, uint32_t itemSizeInU32, gsl::span<uint32_t> outFoundItems)
{
    for (uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; globalThreadIdx < numItems; globalThreadIdx += gridDim.x * blockDim.x) {
        const uint32_t* pItem = &inItems[globalThreadIdx * itemSizeInU32];
        outFoundItems[globalThreadIdx] = hashTable.find(pItem) == Table::not_found ? not_found : 0;
    }
}
template <typename Table>
static __global__ void findAsWarp_kernel(Table hashTable, gsl::span<const uint32_t> inItems, uint32_t numItems, uint32_t itemSizeInU32, gsl::span<uint32_t> outFoundItems)
{
#ifdef __CUDA_ARCH__
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < numItems; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx < numItems) {
            const uint32_t* pItem = &inItems[warpIdx * itemSizeInU32];
            outFoundItems[warpIdx] = hashTable.findAsWarp(pItem) == Table::not_found ? not_found : 0;
        }
    }
#endif
}
template <typename Table>
static __global__ void findAsWarpHybrid_kernel(Table hashTable, gsl::span<const uint32_t> inItems, uint32_t numItems, uint32_t itemSizeInU32, gsl::span<uint32_t> outFoundItems)
{
    const uint32_t gridSize = blockDim.x * gridDim.x;
    const uint32_t end = ((uint32_t)inItems.size() + 31) & (~31);
    for (uint32_t globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x; globalThreadIdx < end; globalThreadIdx += gridSize) {
        check(__activemask() == 0xFFFFFFFF);
        const bool valid = globalThreadIdx < numItems;
        const uint32_t* pItem = valid ? &inItems[globalThreadIdx * itemSizeInU32] : nullptr;
        const uint32_t outHandle = hashTable.findAsWarpHybrid(pItem, valid);
        if (valid) {
            outFoundItems[globalThreadIdx] = outHandle == Table::not_found ? not_found : 0;
        }
    }
}
template <typename Table>
static __global__ void findCountSlotsAsWarp_kernel(Table hashTable, gsl::span<const uint32_t> inItems, uint32_t numItems, uint32_t itemSizeInU32, gsl::span<uint32_t> outSlotsPerItem)
{
#ifdef __CUDA_ARCH__
    const auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
    const auto warpsPerBlock = warp.meta_group_size();
    const auto warpInBlock = warp.meta_group_rank();
    for (uint32_t firstWarpIdx = blockIdx.x * warpsPerBlock; firstWarpIdx < numItems; firstWarpIdx += gridDim.x * warpsPerBlock) {
        const uint32_t warpIdx = firstWarpIdx + warpInBlock;
        if (warpIdx < numItems) {
            const uint32_t* pItem = &inItems[warpIdx * itemSizeInU32];
            outSlotsPerItem[warpIdx] = hashTable.findCountSlotsAsWarp(pItem);
        }
    }
#endif
}

template <typename Table, ThreadType threadTypeAdd_, ThreadType threadTypeFind_>
class MyHashTableWrapper {
public:
    static constexpr ThreadType threadTypeAdd = threadTypeAdd_;
    static constexpr ThreadType threadTypeFind = threadTypeFind_;

    MyHashTableWrapper(const TestSettings& settings)
        : settings(settings)
    {
        hashTable = Table::allocate(settings.numBuckets, settings.numReservedItems, settings.itemSizeInU32);
    }
    ~MyHashTableWrapper()
    {
        hashTable.free();
    }

    void bulkAdd(gsl::span<const uint32_t> inItems, uint32_t numItems)
    {
        const uint32_t threadsPerWorkGroup = settings.warpsPerWorkGroup * threadPerWarp;
        if constexpr (threadTypeAdd == ThreadType::Thread) {
            const uint32_t numWorkGroups = std::min(Utils::divideRoundUp(numItems, threadsPerWorkGroup), settings.maxNumWorkGroups);
            add_kernel<<<numWorkGroups, threadsPerWorkGroup>>>(hashTable, inItems, numItems, settings.itemSizeInU32);
        } else if constexpr (threadTypeAdd == ThreadType::Warp) {
            const uint32_t numWorkGroups = std::min(Utils::divideRoundUp(numItems, settings.warpsPerWorkGroup), settings.maxNumWorkGroups);
            addAsWarp_kernel<<<numWorkGroups, threadsPerWorkGroup>>>(hashTable, inItems, numItems, settings.itemSizeInU32);
        } else if constexpr (threadTypeAdd == ThreadType::WarpHybrid) {
            const uint32_t numWorkGroups = std::min(Utils::divideRoundUp(numItems + threadPerWarp, threadsPerWorkGroup), settings.maxNumWorkGroups);
            addAsWarpHybrid_kernel<<<numWorkGroups, threadsPerWorkGroup>>>(hashTable, inItems, numItems, settings.itemSizeInU32);
        } else {
            checkAlways(false);
        }
    }
    void bulkFind(gsl::span<const uint32_t> inItems, gsl::span<uint32_t> outFoundItems)
    {
        const uint32_t threadsPerWorkGroup = settings.warpsPerWorkGroup * threadPerWarp;
        const uint32_t numItems = (uint32_t)outFoundItems.size();
        if constexpr (threadTypeFind == ThreadType::Thread) {
            const uint32_t numWorkGroups = std::min(Utils::divideRoundUp(numItems, threadsPerWorkGroup), settings.maxNumWorkGroups);
            find_kernel<<<numWorkGroups, threadsPerWorkGroup>>>(hashTable, inItems, numItems, settings.itemSizeInU32, outFoundItems);
        } else if constexpr (threadTypeFind == ThreadType::Warp) {
            const uint32_t numWorkGroups = std::min(Utils::divideRoundUp(numItems, settings.warpsPerWorkGroup), settings.maxNumWorkGroups);
            findAsWarp_kernel<<<numWorkGroups, threadsPerWorkGroup>>>(hashTable, inItems, numItems, settings.itemSizeInU32, outFoundItems);
        } else if constexpr (threadTypeFind == ThreadType::WarpHybrid) {
            const uint32_t numWorkGroups = std::min(Utils::divideRoundUp(numItems + threadPerWarp, threadsPerWorkGroup), settings.maxNumWorkGroups);
            findAsWarpHybrid_kernel<<<numWorkGroups, threadsPerWorkGroup>>>(hashTable, inItems, numItems, settings.itemSizeInU32, outFoundItems);
        } else {
            checkAlways(false);
        }
    }
    void bulkFindCountSlots(gsl::span<const uint32_t> inItems, gsl::span<uint32_t> outSlotsPerItem)
    {
        const uint32_t threadsPerWorkGroup = settings.warpsPerWorkGroup * threadPerWarp;
        const uint32_t numItems = (uint32_t)outSlotsPerItem.size();
        if constexpr (threadTypeFind == ThreadType::Warp) {
            const uint32_t numWorkGroups = std::min(Utils::divideRoundUp(numItems, settings.warpsPerWorkGroup), settings.maxNumWorkGroups);
            findCountSlotsAsWarp_kernel<<<numWorkGroups, threadsPerWorkGroup>>>(hashTable, inItems, numItems, settings.itemSizeInU32, outSlotsPerItem);
        } else {
            checkAlways(false);
        }
    }

    my_units::bytes memory_used_by_items() const { return hashTable.memory_used_by_items(); }
    my_units::bytes memory_used_by_slabs() const { return hashTable.memory_used_by_slabs(); }
    my_units::bytes memory_allocated() const { return hashTable.memory_allocated(); }

private:
    TestSettings settings;
    Table hashTable;
};

class SlabHashWrapper {
public:
    static constexpr ThreadType threadTypeAdd = ThreadType::WarpHybrid;
    static constexpr ThreadType threadTypeFind = ThreadType::WarpHybrid;

public:
    SlabHashWrapper(const TestSettings& settings)
        : itemSizeInU32(settings.itemSizeInU32)
        , hashTable(gpu_hash_table<uint32_t, uint32_t, SlabHashTypeT::ConcurrentSet>(settings.numReservedItems, settings.numBuckets, 0, 123456789, false))
    {
        checkAlways(itemSizeInU32 == 1);
    }
    SlabHashWrapper(SlabHashWrapper&&) = delete;
    ~SlabHashWrapper() = default;

    void bulkAdd(gsl::span<const uint32_t> inItems, uint32_t numItems)
    {
        hashTable.hash_build_device((uint32_t*)inItems.data(), nullptr, numItems);
    }
    void bulkFind(gsl::span<const uint32_t> inItems, gsl::span<uint32_t> outFoundItems)
    {
        const uint32_t numItems = (uint32_t)inItems.size() / itemSizeInU32;
        hashTable.hash_search_device((uint32_t*)inItems.data(), outFoundItems.data(), numItems);
    }

private:
    uint32_t itemSizeInU32;
    gpu_hash_table<uint32_t, uint32_t, SlabHashTypeT::ConcurrentSet> hashTable;
};
template <typename key_t>
class DyCuckooWrapper {
public:
    static constexpr ThreadType threadTypeAdd = ThreadType::WarpHybrid;
    static constexpr ThreadType threadTypeFind = ThreadType::WarpHybrid;

public:
    DyCuckooWrapper(const TestSettings& settings)
        : hashTable(settings.numReservedItems, 1000000, 0.6, 0.9)
    {
        checkAlways(settings.itemSizeInU32 * sizeof(uint32_t) == sizeof(key_t));
    }
    DyCuckooWrapper(DyCuckooWrapper&&) = delete;
    ~DyCuckooWrapper() = default;

    void bulkAdd(gsl::span<const uint32_t> items, uint32_t numItems)
    {
        hashTable.batch_insert((key_t*)items.data(), numItems);
    }
    void bulkFind(gsl::span<const uint32_t> items, gsl::span<uint32_t> outItems)
    {
        hashTable.batch_search((key_t*)items.data(), outItems.data(), (uint32_t)outItems.size());
    }

private:
    DynamicCuckoo<key_t> hashTable;
};
template <typename key_t>
class DyCuckooStaticWrapper {
public:
    static constexpr ThreadType threadTypeAdd = ThreadType::WarpHybrid;
    static constexpr ThreadType threadTypeFind = ThreadType::WarpHybrid;

public:
    DyCuckooStaticWrapper(const TestSettings& settings)
        : hashTable(settings.numReservedItems)
    {
        checkAlways(settings.itemSizeInU32 * sizeof(uint32_t) == sizeof(key_t));
    }
    DyCuckooStaticWrapper(DyCuckooStaticWrapper&&) = delete;
    ~DyCuckooStaticWrapper() = default;

    void bulkAdd(gsl::span<const uint32_t> items, uint32_t numItems)
    {
        hashTable.batch_insert((key_t*)items.data(), numItems);
    }
    void bulkFind(gsl::span<const uint32_t> items, gsl::span<uint32_t> outItems)
    {
        hashTable.batch_search((key_t*)items.data(), outItems.data(), (uint32_t)outItems.size());
    }

private:
    StaticCuckoo<key_t> hashTable;
};

struct TestResults {
    std::chrono::duration<double> bulkAdd1;
    std::chrono::duration<double> bulkAdd2;
    std::chrono::duration<double> bulkSearch_hit0;
    std::chrono::duration<double> bulkSearch_hit25;
    std::chrono::duration<double> bulkSearch_hit50;
    std::chrono::duration<double> bulkSearch_hit75;
    std::chrono::duration<double> bulkSearch_hit100;

    // Memory statistics after first insertion.
    size_t memoryUsedByItems1 = 0, memoryUsedBySlabs1 = 0, memoryAllocated1 = 0;
    // Memory statistics after second insertion.
    size_t memoryUsedByItems2 = 0, memoryUsedBySlabs2 = 0, memoryAllocated2 = 0;

    void print(const std::string& name, const TestSettings& settings) const
    {
        printf("=====================================================\n");
        printf(" %s\n", name.c_str());
        printf("=====================================================\n");
        printf("Memory (items):   [%8u]: %.3f MB\n", settings.numInitialItems, Utils::to_MB(memoryUsedByItems1));
        printf("Memory (slabs):   [%8u]: %.3f MB\n", settings.numInitialItems, Utils::to_MB(memoryUsedBySlabs1));
        printf("Initial insertion [%8u]: %.3f ms\n", settings.numInitialItems, std::chrono::duration<double, std::milli>(bulkAdd1).count());
        printf("Second insertion  [%8u]: %.3f ms\n", settings.numInsertItems, std::chrono::duration<double, std::milli>(bulkAdd2).count());
        printf("Search 0          [%8u]: %.3f ms\n", settings.numSearchItems, std::chrono::duration<double, std::milli>(bulkSearch_hit0).count());
        printf("Search 25         [%8u]: %.3f ms\n", settings.numSearchItems, std::chrono::duration<double, std::milli>(bulkSearch_hit25).count());
        printf("Search 50         [%8u]: %.3f ms\n", settings.numSearchItems, std::chrono::duration<double, std::milli>(bulkSearch_hit50).count());
        printf("Search 75         [%8u]: %.3f ms\n", settings.numSearchItems, std::chrono::duration<double, std::milli>(bulkSearch_hit75).count());
        printf("Search 100        [%8u]: %.3f ms\n", settings.numSearchItems, std::chrono::duration<double, std::milli>(bulkSearch_hit100).count());
        printf("=====================================================\n\n");
    }

    nlohmann::json toJson() const
    {
        nlohmann::json out {};
        out["bulkAdd1_ms"] = std::chrono::duration<double, std::milli>(bulkAdd1).count();
        out["bulkAdd2_ms"] = std::chrono::duration<double, std::milli>(bulkAdd2).count();
        out["bulkSearch_hit0_ms"] = std::chrono::duration<double, std::milli>(bulkSearch_hit0).count();
        out["bulkSearch_hit25_ms"] = std::chrono::duration<double, std::milli>(bulkSearch_hit25).count();
        out["bulkSearch_hit50_ms"] = std::chrono::duration<double, std::milli>(bulkSearch_hit50).count();
        out["bulkSearch_hit75_ms"] = std::chrono::duration<double, std::milli>(bulkSearch_hit75).count();
        out["bulkSearch_hit100_ms"] = std::chrono::duration<double, std::milli>(bulkSearch_hit100).count();

        out["insert1_memory_used_by_items_bytes"] = memoryUsedByItems1;
        out["insert1_memory_used_by_slabs_bytes"] = memoryUsedBySlabs1;
        out["insert1_memory_allocated_bytes"] = memoryAllocated1;

        out["insert2_memory_used_by_items_bytes"] = memoryUsedByItems2;
        out["insert2_memory_used_by_slabs_bytes"] = memoryUsedBySlabs2;
        out["insert2_memory_allocated_bytes"] = memoryAllocated2;
        return out;
    }
};

static std::vector<uint32_t> generateRandomItems(std::default_random_engine& re, uint32_t numItems, uint32_t itemSizeInU32)
{
    PROFILE_FUNCTION();
    std::vector<uint32_t> items;
    items.resize(numItems * itemSizeInU32);
    std::uniform_int_distribution<uint32_t> distribution;
    for (uint32_t& item : items)
        item = distribution(re);
    return items;
}
static std::vector<uint32_t> shuffleItems(std::default_random_engine& re, gsl::span<const uint32_t> items, uint32_t itemSizeInU32)
{
    PROFILE_FUNCTION();
    checkAlways(items.size() % itemSizeInU32 == 0);

    const size_t numItems = items.size() / itemSizeInU32;
    std::vector<uint32_t> indices(numItems);
    std::iota(std::begin(indices), std::end(indices), 0);
    std::shuffle(std::begin(indices), std::end(indices), re);

    std::vector<uint32_t> out(items.size());
    for (size_t i = 0; i < numItems; ++i) {
        const uint32_t j = indices[i] * itemSizeInU32;
        const uint32_t k = i * itemSizeInU32;
        for (uint32_t l = 0; l < itemSizeInU32; ++l) {
            out[k + l] = items[j + l];
        }
    }
    return out;
}

template <typename T>
concept findExternalTimer = requires(T hashTable, gsl::span<const uint32_t> inItems, gsl::span<uint32_t> outItems) {
    hashTable.bulkFind(inItems, outItems);
};
static_assert(findExternalTimer<MyHashTableWrapper<TicketBoardHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>);

template <typename T>
void runFind(T& hashTable, gsl::span<const uint32_t> inItems, gsl::span<uint32_t> outItems)
{
    PROFILE_FUNCTION();
    if constexpr (findExternalTimer<T>) {
        hashTable.bulkFind(inItems, outItems);
    } else {
        const auto runner = [&](auto&& f) {
            f();
        };
        hashTable.bulkFind(inItems, outItems, runner);
    }
}
template <typename T>
void runFind(T& hashTable, gsl::span<const uint32_t> inItems, gsl::span<uint32_t> outItems, GPUTimingsManager& timings, std::string timingName)
{
    if constexpr (findExternalTimer<T>) {
        timings.startTiming(timingName, nullptr);
        hashTable.bulkFind(inItems, outItems);
        timings.endTiming(timingName, nullptr);
    } else {
        const auto runner = [&](auto&& f) {
            timings.startTiming(timingName, nullptr);
            f();
            timings.endTiming(timingName, nullptr);
        };
        hashTable.bulkFind(inItems, outItems, runner);
    }
}

class DataCache {
public:
    std::vector<uint32_t>& getInitialItems(const TestSettings& settings)
    {
        if (m_initialItems.empty())
            m_initialItems = generateRandomItems(re, settings.numInitialItems, settings.itemSizeInU32);
        return m_initialItems;
    }

    std::vector<uint32_t>& getInsertItems(const TestSettings& settings)
    {
        if (m_insertItems.empty())
            m_insertItems = generateRandomItems(re, settings.numInsertItems, settings.itemSizeInU32);
        return m_insertItems;
    }

    std::vector<uint32_t>& getNonInsertItems(const TestSettings& settings)
    {
        if (m_nonInsertItems.empty())
            m_nonInsertItems = generateRandomItems(re, settings.numSearchItems, settings.itemSizeInU32);
        return m_nonInsertItems;
    }

    std::vector<uint32_t>& getSearchItems(const TestSettings& settings, int iHitRatePercentage)
    {
        if (auto iter = m_searchItems.find(iHitRatePercentage); iter != std::end(m_searchItems)) {
            return iter->second;
        } else {
            const uint32_t numUnfindableItems = settings.numSearchItems * (100 - iHitRatePercentage) / 100;
            auto searchItems = generateRandomItems(re, numUnfindableItems, settings.itemSizeInU32);

            // Random shuffle so we don't query in the order that we inserted.
            const auto shuffledInitialItems = shuffleItems(re, getInitialItems(settings), settings.itemSizeInU32);

            const auto tmp = (settings.numSearchItems - numUnfindableItems) * settings.itemSizeInU32;
            searchItems.reserve(searchItems.size() + tmp);
            std::copy_n(std::begin(shuffledInitialItems), tmp, std::back_inserter(searchItems));
            //searchItems = shuffleItems(re, searchItems, settings.itemSizeInU32); // Make sure that hash table hits/misses are in a random order.
            return m_searchItems[iHitRatePercentage] = searchItems;
        }
    }

private:
    std::default_random_engine re {};
    std::vector<uint32_t> m_initialItems;
    std::vector<uint32_t> m_insertItems;
    std::vector<uint32_t> m_nonInsertItems;
    std::unordered_map<int, std::vector<uint32_t>> m_searchItems;
};

template <typename Table>
static void countSlotsPerSearchOperation(TestSettings settings, DataCache& inOutDataCache)
{
    PROFILE_FUNCTION();

    Table table { settings };

    const cudaStream_t cudaStream = nullptr;
    GPUTimingsManager timings {};

    // Measure initial insertion performance.
    if (settings.numInitialItems > 0) {
        PROFILE_SCOPE("bulkAdd1");
        auto initialItemsGPU = StaticArray<uint32_t>::allocate("Items", inOutDataCache.getInitialItems(settings), EMemoryType::GPU_Malloc);
        checkAlways(initialItemsGPU.size() == settings.numInitialItems * settings.itemSizeInU32);

        timings.startTiming("bulkAdd1", cudaStream);
        table.bulkAdd(initialItemsGPU, settings.numInitialItems);
        timings.endTiming("bulkAdd1", cudaStream);
        CUDA_CHECK_ERROR();

        initialItemsGPU.free();
    }

    // Measure search performance.
    const auto measureSearchPerformance = [&](int iHitRatePercentage) {
        PROFILE_SCOPE("measureSearchPerformance");

        auto searchItemsGPU = StaticArray<uint32_t>::allocate("searchItemsCPU", inOutDataCache.getSearchItems(settings, iHitRatePercentage), EMemoryType::GPU_Malloc);
        auto slotsPerItemGPU = StaticArray<uint32_t>::allocate("slotsPerItemGPU", settings.numSearchItems, EMemoryType::GPU_Malloc);
        cudaMemset(slotsPerItemGPU.data(), 0xFF, slotsPerItemGPU.size_in_bytes());

        const auto timingName = "search_hit" + std::to_string(iHitRatePercentage);
        table.bulkFindCountSlots(searchItemsGPU, slotsPerItemGPU);
        CUDA_CHECK_ERROR();

        const auto slotsPerItem = slotsPerItemGPU.copy_to_cpu();
        const uint32_t totalChecks = std::reduce(std::execution::par_unseq, std::begin(slotsPerItem), std::end(slotsPerItem), 0, std::plus<uint32_t>());
        printf("Checks per slot: %f\n", (double)totalChecks / (double)slotsPerItem.size());

        searchItemsGPU.free();
        slotsPerItemGPU.free();
    };
    measureSearchPerformance(0);
    measureSearchPerformance(25);
    measureSearchPerformance(50);
    measureSearchPerformance(75);
    measureSearchPerformance(100);
}

template <typename Table>
static void runTest(TestSettings settings, TestResults& out, DataCache& inOutDataCache)
{
    PROFILE_FUNCTION();

    Table table { settings };

    const cudaStream_t cudaStream = nullptr;
    GPUTimingsManager timings {};

    // Measure initial insertion performance.
    if (settings.numInitialItems > 0) {
        PROFILE_SCOPE("bulkAdd1");
        auto initialItemsGPU = StaticArray<uint32_t>::allocate("Items", inOutDataCache.getInitialItems(settings), EMemoryType::GPU_Malloc);
        checkAlways(initialItemsGPU.size() == settings.numInitialItems * settings.itemSizeInU32);

        timings.startTiming("bulkAdd1", cudaStream);
        table.bulkAdd(initialItemsGPU, settings.numInitialItems);
        timings.endTiming("bulkAdd1", cudaStream);
        CUDA_CHECK_ERROR();

        initialItemsGPU.free();
    }

#if CAPTURE_MEMORY_STATS_SLOW
    cudaDeviceSynchronize();
    if constexpr (requires(Table t) { t.memory_used_by_items(); }) {
        out.memoryUsedByItems1 = table.memory_used_by_items();
        out.memoryUsedBySlabs1 = table.memory_used_by_slabs();
        out.memoryAllocated1 = table.memory_allocated();
    }
#endif

    // Measure search performance.
    const auto measureSearchPerformance = [&](int iHitRatePercentage) {
        PROFILE_SCOPE("measureSearchPerformance");

        auto searchItemsGPU = StaticArray<uint32_t>::allocate("searchItemsCPU", inOutDataCache.getSearchItems(settings, iHitRatePercentage), EMemoryType::GPU_Malloc);
        auto itemsFoundGPU = StaticArray<uint32_t>::allocate("itemsFoundGPU", settings.numSearchItems, EMemoryType::GPU_Malloc);
        cudaMemset(itemsFoundGPU.data(), 0xFF, itemsFoundGPU.size_in_bytes());

        const auto timingName = "search_hit" + std::to_string(iHitRatePercentage);
        runFind(table, searchItemsGPU, itemsFoundGPU, timings, timingName);
        CUDA_CHECK_ERROR();

#if CHECK_OUTPUT_FOR_CORRECTNESS
        auto itemsFoundCPU = itemsFoundGPU.copy_to_cpu();
        const uint32_t numFound = std::transform_reduce(
            std::execution::par_unseq,
            std::begin(itemsFoundCPU),
            std::end(itemsFoundCPU),
            0u,
            std::plus<uint32_t>(),
            [](uint32_t v) { return (v != not_found ? 1 : 0); });
        const double percentage = (double)numFound / (double)itemsFoundGPU.size() * 100.0;
#ifndef __CUDA_ARCH__
        checkAlways(searchItemsGPU.size() == settings.numSearchItems * settings.itemSizeInU32);
        checkAlways(percentage > (iHitRatePercentage - 1));
        checkAlways(percentage < (iHitRatePercentage + 5));
#endif // __CUDA_ARCH__
#endif // CHECK_OUTPUT_FOR_CORRECTNESS

        searchItemsGPU.free();
        itemsFoundGPU.free();
    };
    measureSearchPerformance(0);
    measureSearchPerformance(25);
    measureSearchPerformance(50);
    measureSearchPerformance(75);
    measureSearchPerformance(100);

    // Measure second insertion performance.
    if (settings.numInsertItems > 0) {
        PROFILE_SCOPE("bulkAdd2");
        auto insertItemsGPU = StaticArray<uint32_t>::allocate("Insert Items", inOutDataCache.getInsertItems(settings), EMemoryType::GPU_Malloc);

        timings.startTiming("bulkAdd2", cudaStream);
        table.bulkAdd(insertItemsGPU, settings.numInsertItems);
        timings.endTiming("bulkAdd2", cudaStream);
        CUDA_CHECK_ERROR();

        // Verify insertions.
#if CHECK_OUTPUT_FOR_CORRECTNESS
        {
            auto itemsFoundGPU = StaticArray<uint32_t>::allocate("Verify items", settings.numInsertItems, EMemoryType::GPU_Malloc);
            cudaMemset(itemsFoundGPU.data(), 0, itemsFoundGPU.size_in_bytes());
            runFind(table, insertItemsGPU, itemsFoundGPU);
            CUDA_CHECK_ERROR();

            {
                std::vector<uint32_t> itemsFoundCPU = itemsFoundGPU.copy_to_cpu();
                std::for_each(
                    std::execution::par_unseq,
                    std::begin(itemsFoundCPU),
                    std::end(itemsFoundCPU),
                    [](uint32_t v) {
#ifndef __CUDA_ARCH__
                        checkAlways(v != not_found);
#endif
                    });
            }

            auto nonInsertedItemsGPU = StaticArray<uint32_t>::allocate("Verify items", inOutDataCache.getNonInsertItems(settings), EMemoryType::GPU_Malloc);
            checkAlways(nonInsertedItemsGPU.size() == itemsFoundGPU.size() * settings.itemSizeInU32);

            cudaMemset(itemsFoundGPU.data(), 0xFF, itemsFoundGPU.size_in_bytes());
            runFind(table, nonInsertedItemsGPU, itemsFoundGPU);
            CUDA_CHECK_ERROR();

            {
                std::vector<uint32_t> itemsFoundCPU = itemsFoundGPU.copy_to_cpu();
                const uint32_t foundDummyValues = std::transform_reduce(
                    std::execution::par_unseq,
                    std::begin(itemsFoundCPU),
                    std::end(itemsFoundCPU),
                    0u,
                    std::plus<uint32_t>(),
                    [](uint32_t v) { return (v != not_found ? 1 : 0); });

                // These values should not have been inserted. The RNG may have generated the same numbers
                // multiple times though so test that the number of hits is very low (rather than exactly 0).
                // printf("Hit percentage dummy: %u out of %u\n", foundDummyValues, numDummyItems);
                checkAlways(foundDummyValues < itemsFoundGPU.size() / 20);
            }
            nonInsertedItemsGPU.free();

            itemsFoundGPU.free();
        }
#endif

        insertItemsGPU.free();
    }

#if CAPTURE_MEMORY_STATS_SLOW
    cudaDeviceSynchronize();
    if constexpr (requires(Table t) { t.memory_used_by_items(); }) {
        out.memoryUsedByItems2 = table.memory_used_by_items();
        out.memoryUsedBySlabs2 = table.memory_used_by_slabs();
        out.memoryAllocated2 = table.memory_allocated();
    }
#endif

    out.bulkAdd1 = timings.getBlocking("bulkAdd1");
    out.bulkAdd2 = timings.getBlocking("bulkAdd2");
    out.bulkSearch_hit0 = timings.getBlocking("search_hit0");
    out.bulkSearch_hit25 = timings.getBlocking("search_hit25");
    out.bulkSearch_hit50 = timings.getBlocking("search_hit50");
    out.bulkSearch_hit75 = timings.getBlocking("search_hit75");
    out.bulkSearch_hit100 = timings.getBlocking("search_hit100");
}

template <typename T>
void runTestAndPrint(std::string name, const TestSettings& settings, DataCache& inOutDataCache)
{
    TestResults testResults {};
    runTest<T>(settings, testResults, inOutDataCache);
    testResults.print(name, settings);
};

template <typename T>
nlohmann::json runTestJson(std::string name, const TestSettings& settings, DataCache& inOutDataCache)
{
    TestResults testResults {};
    // First run to capture memory statistics.
    // Memory statistics may be slow to compute and impact GPU performance (clocks, etc).
    printf("Testing %s\n", name.c_str());
    runTest<T>(settings, testResults, inOutDataCache);

    nlohmann::json out {};
    out["hash_table"] = name;
#ifndef __CUDA_ARCH__
    out["thread_type_add"] = toString(T::threadTypeAdd);
    out["thread_type_find"] = toString(T::threadTypeFind);
#endif
    out["defines"] = getDefineInfoJson();
    out["system"] = getSystemInfoJson();
    out["settings"] = settings.toJson();
    out["results"] = testResults.toJson();
    return out;
};

static TestSettings parseCommandLineArguments(int argc, char** argv);

int main(int argc, char** argv)
{
    TestSettings settings = parseCommandLineArguments(argc, argv);
    settings.print();
    printf("\n\n");

    DataCache dataCache;
    if (settings.optOutFilePath) {
        std::vector<nlohmann::json> outVec;
        for (uint32_t run = 0; run < settings.numRuns; ++run) {
            if (settings.itemSizeInU32 == 1) {
                outVec.push_back(runTestJson<DyCuckooWrapper<uint32_t>>("DyCuckoo", settings, dataCache));
                outVec.push_back(runTestJson<DyCuckooStaticWrapper<uint32_t>>("DyCuckooStatic", settings, dataCache));
                outVec.push_back(runTestJson<SlabHashWrapper>("SlabHash", settings, dataCache));
            } else {
                if (settings.itemSizeInU32 == 2) {
                    outVec.push_back(runTestJson<DyCuckooWrapper<uint64_t>>("DyCuckoo", settings, dataCache));
                    outVec.push_back(runTestJson<DyCuckooStaticWrapper<uint64_t>>("DyCuckooStatic", settings, dataCache));
                }

                outVec.push_back(runTestJson<MyHashTableWrapper<Atomic64HashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>("Atomic64HashTable", settings, dataCache));
                outVec.push_back(runTestJson<MyHashTableWrapper<AccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>("AccelerationHashTable", settings, dataCache));
                outVec.push_back(runTestJson<MyHashTableWrapper<CompactAccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>("CompactAccelerationHashTable", settings, dataCache));
                outVec.push_back(runTestJson<MyHashTableWrapper<TicketBoardHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>("TicketBoardHashTable", settings, dataCache));

                outVec.push_back(runTestJson<MyHashTableWrapper<Atomic64HashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("Atomic64HashTable", settings, dataCache));
                outVec.push_back(runTestJson<MyHashTableWrapper<AccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("AccelerationHashTable", settings, dataCache));
                outVec.push_back(runTestJson<MyHashTableWrapper<CompactAccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("CompactAccelerationHashTable", settings, dataCache));
                outVec.push_back(runTestJson<MyHashTableWrapper<TicketBoardHashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("TicketBoardHashTable", settings, dataCache));
            }
        }

        nlohmann::json system;
        system["name"] = getHostNameCpp();
        system["os"] = getOperatingSystemName();

        nlohmann::json out;
        out["results"] = outVec;
        out["system"] = system;
        std::ofstream outFile { *settings.optOutFilePath };
        outFile << std::setw(4) << nlohmann::json(outVec);
    } else {
        for (uint32_t run = 0; run < settings.numRuns; ++run) {
            if (settings.itemSizeInU32 == 1) {
                runTestAndPrint<MyHashTableWrapper<AccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("AccelerationHashTable", settings, dataCache);
                runTestAndPrint<SlabHashWrapper>("SlabHash", settings, dataCache);
                // runTestAndPrint<DyCuckooWrapper<uint32_t>>("DyCuckoo", settings, dataCache);
                // runTestAndPrint<DyCuckooStaticWrapper<uint32_t>>("DyCuckooStatic", settings, dataCache);
            } else {
                if (settings.itemSizeInU32 == 2) {
                    runTestAndPrint<DyCuckooWrapper<uint64_t>>("DyCuckoo", settings, dataCache);
                    runTestAndPrint<DyCuckooStaticWrapper<uint64_t>>("DyCuckooStatic", settings, dataCache);
                }

                /*printf("32-bit ACCELERATION HASH: ");
                countSlotsPerSearchOperation<MyHashTableWrapper<AccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>(settings, dataCache);

                printf("\n8-bit ACCELERATION HASH: ");
                countSlotsPerSearchOperation<MyHashTableWrapper<CompactAccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>(settings, dataCache);*/

                runTestAndPrint<MyHashTableWrapper<Atomic64HashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>("Atomic64HashTable", settings, dataCache);
                runTestAndPrint<MyHashTableWrapper<AccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>("AccelerationHashTable", settings, dataCache);
                runTestAndPrint<MyHashTableWrapper<CompactAccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>("CompactAccelerationHashTable", settings, dataCache);
                runTestAndPrint<MyHashTableWrapper<TicketBoardHashTable<EMemoryType::GPU_Malloc>, ThreadType::Warp, ThreadType::Warp>>("TicketBoardHashTable", settings, dataCache);

                runTestAndPrint<MyHashTableWrapper<Atomic64HashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("Atomic64HashTable", settings, dataCache);
                runTestAndPrint<MyHashTableWrapper<AccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("AccelerationHashTable", settings, dataCache);
                runTestAndPrint<MyHashTableWrapper<CompactAccelerationHashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("CompactAccelerationHashTable", settings, dataCache);
                runTestAndPrint<MyHashTableWrapper<TicketBoardHashTable<EMemoryType::GPU_Malloc>, ThreadType::WarpHybrid, ThreadType::WarpHybrid>>("TicketBoardHashTable", settings, dataCache);

                runTestAndPrint<MyHashTableWrapper<TicketBoardHashTable<EMemoryType::GPU_Malloc>, ThreadType::Thread, ThreadType::Thread>>("TicketBoardHashTable", settings, dataCache);
            }
        }
    }
    return 0;
}

static TestSettings parseCommandLineArguments(int argc, char** argv)
{
    constexpr float targetLoadFactor = 64;
    constexpr uint32_t numInitialItems = 8 * 1024 * 1024;
    constexpr uint32_t numInsertItems = 1024 * 1024;
    constexpr uint32_t numSearchItems = 1024 * 1024;
    // constexpr uint32_t numInitialItems = 1024;
    // constexpr uint32_t numInsertItems = 128;
    // constexpr uint32_t numSearchItems = 512;
    constexpr uint32_t numItems = numInitialItems + numInsertItems;

    TestSettings out {
        .itemSizeInU32 = 2,
        .numInitialItems = numInitialItems,
        .numInsertItems = numInsertItems,
        .numReservedItems = numItems * 2,
        .numSearchItems = numSearchItems,
        .numBuckets = (uint32_t)(numItems / targetLoadFactor) + 1u,
        .warpsPerWorkGroup = 2,
        .maxNumWorkGroups = 16384,
        .numRuns = 1
    };

    CLI::App app { "HashMap Benchmark" };
    app.add_option("--item_size", out.itemSizeInU32, "Item size measured in sizeof(U32)");
    app.add_option("--num_initial_items", out.numInitialItems, "Number of items to insert into (empty) hash map");
    app.add_option("--num_insert_items", out.numInsertItems, "Number of items to insert into (empty) hash map");
    app.add_option("--num_reserved_items", out.numReservedItems, "Number of items reserved by the memory allocator");
    app.add_option("--num_search_items", out.numSearchItems, "Number of items to search");
    app.add_option("--num_buckets", out.numBuckets, "Number of buckets (size of hash table)");
    app.add_option("--warps_per_work_group", out.warpsPerWorkGroup, "Number of warps (32 threads) per work group");
    app.add_option("--max_num_work_groups", out.maxNumWorkGroups, "Maximum number of work groups");
    app.add_option("--runs", out.numRuns, "Number of benchmark runs");
    app.add_option("--out", out.optOutFilePath, "Path of JSON output file");
    try {
        app.parse(argc, argv);
    } catch (const std::exception& e) {
        printf("Exception: %s\n", e.what());
        exit(1);
    }

    return out;
}
