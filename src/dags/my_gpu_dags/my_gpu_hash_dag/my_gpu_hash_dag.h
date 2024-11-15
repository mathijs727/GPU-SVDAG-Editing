#pragma once
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/ticket_board_hash_table.h"
// ^^^ Must be included before typedefs.h to prevent link error on Windows ^^^

#include "array.h"
#include "array2d.h"
#include "configuration/gpu_hash_dag_definitions.h"
#include "configuration/hash_dag_enum.h"
#include "cuda_helpers_cpp.h"
#include "dags/hash_dag/hash_dag_globals.h"
#include "dags/my_gpu_dags/my_gpu_base_dag.h"
#include "dags/my_gpu_dags/my_gpu_dag_item.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/atomic64_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/compact_acceleration_hash_table.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/hash_tables/individual_chaining_hash_table.h"
#include "typedefs.h"
#include "utils.h"
#include <array>

class StatsRecorder;

template <HashTableType, EMemoryType>
struct hash_table_selector;
template <EMemoryType MemoryType>
struct hash_table_selector<HashTableType::Atomic64, MemoryType> {
    using type = Atomic64HashTable<MemoryType>;
};
template <EMemoryType MemoryType>
struct hash_table_selector<HashTableType::TicketBoard, MemoryType> {
    using type = TicketBoardHashTable<MemoryType>;
};
template <EMemoryType MemoryType>
struct hash_table_selector<HashTableType::AccelerationHash, MemoryType> {
    using type = AccelerationHashTable<MemoryType>;
};
template <EMemoryType MemoryType>
struct hash_table_selector<HashTableType::CompactAccelerationHash, MemoryType> {
    using type = CompactAccelerationHashTable<MemoryType>;
};
template <EMemoryType MemoryType>
struct hash_table_selector<HashTableType::IndividualChaining, MemoryType> {
    using type = IndividualChainingHashTable<MemoryType>;
};
template <HashTableType HashTableType, EMemoryType MemoryType>
using hash_table_selector_t = typename hash_table_selector<HashTableType, MemoryType>::type;

// clang-format off
template <typename T>
concept has_add_as_warp = requires(T& container, const uint32_t* pItem) {
    { container.addAsWarp(pItem) } -> std::same_as<uint32_t>;
};
template <typename T>
concept has_verify_no_duplicates = requires(const T& container) {
    { container.verifyNoDuplicates() } -> std::same_as<void>;
};
// clang-format on

template <EMemoryType MemoryType>
#if EDITS_ENABLE_MATERIALS
struct MyGPUHashDAG : public GPUBaseDAG<MyGPUHashDAG<MemoryType>, HASH_DAG_MATERIAL_BITS> {
#else
struct MyGPUHashDAG : public GPUBaseDAG<MyGPUHashDAG<MemoryType>, 0> {
#endif
private:
    static constexpr int version = 3; // Make it easier to decipher stats files.

    using HashTableImpl = hash_table_selector_t<HASH_TABLE_TYPE, MemoryType>;

public:
    StaticArray2D<uint32_t> fullyFilledNodes; // Handles to fully filled nodes for given level & material id.
    uint32_t firstNodeIndex = 0; // Root node

#if EDITS_ENABLE_MATERIALS
    using Super = GPUBaseDAG<MyGPUHashDAG<MemoryType>, HASH_DAG_MATERIAL_BITS>;
#else
    using Super = GPUBaseDAG<MyGPUHashDAG<MemoryType>, 0>;
#endif

    using LeafBuilder = typename Super::LeafBuilder_;

    using NodeDecoder = typename HashTableImpl::ElementDecoder;
    using LeafDecoder = typename HashTableImpl::ElementDecoder;

    using Super::maxItemSizeInU32;
    using Super::maxLeafSizeInU32;
    using Super::maxNodeSizeInU32;
    using Super::minLeafSizeInU32;
    using Super::minNodeSizeInU32;
    using Super::NumMaterials;

public:
    void checkHashTables() const
    {
#ifdef _WIN32
        // Required on Windows before accessing shared memory (
        cudaDeviceSynchronize();
#endif

        if constexpr (has_verify_no_duplicates<HashTableImpl>) {
            for (const auto& hashTable : hashTables)
                hashTable.verifyNoDuplicates();
        }
    }

    template <EMemoryType NewMemoryType>
    MyGPUHashDAG<NewMemoryType> copy() const
    {
        MyGPUHashDAG<NewMemoryType> out;
        // Keep a copy of the HashTableImpl classes on both the CPU & GPU.
        // NOTE(MAthijs): CUDA managed memory would be nice here; but I don't trust it to not tank performance (on Windows).
        out.hashTables = decltype(out.hashTables)::allocate(
            "MyGPUHashDAG::hashTables", Super::maxItemSizeInU32 - Super::minItemSizeInU32 + 1, NewMemoryType);
        std::vector<hash_table_selector_t<HASH_TABLE_TYPE, NewMemoryType>> newHashTables;

        for (const auto& hashtable : hashTables.copy_to_cpu()) {
            newHashTables.push_back(hashtable.template copy<NewMemoryType>());
        }
        if constexpr (MemoryType != EMemoryType::CPU) {
            memcpy(out.hashTables.data(), newHashTables.data(), out.hashTables.size_in_bytes());
        } else {
            cudaMemcpy(out.hashTables.data(), newHashTables.data(), out.hashTables.size_in_bytes(), cudaMemcpyHostToDevice);
        }
        out.fullyFilledNodes = fullyFilledNodes.copy(NewMemoryType);
        out.firstNodeIndex = firstNodeIndex;
        return out;
    }

    static MyGPUHashDAG allocate(uint32_t hashTableSizes);
    static MyGPUHashDAG allocate(gsl::span<const uint32_t> tableSizes);
    void free();

    my_units::bytes memory_allocated() const;
    my_units::bytes memory_used_by_items() const;
    my_units::bytes memory_used_by_slabs() const;

    double getAverageLoadFactor() const;
    void setLoadFactor(double loadFactor);

    void writeTo(BinaryWriter& writer) const;
    void readFrom(BinaryReader& reader);

    HOST_DEVICE bool is_leaf_fully_filled(const ElementEncoder auto pLeaf, uint32_t& outMaterial) const
    {
        if (pLeaf[0] != 0xFFFFFFFF || pLeaf[1] != 0xFFFFFFFF)
            return false;

        if constexpr (Super::MaterialBits > 0) {

            static_assert(Super::MaterialBits < 32);
            constexpr uint32_t MaterialMask = (1u << Super::MaterialBits) - 1u;
            const uint32_t firstMaterial = pLeaf[2] & MaterialMask;

            const uint32_t fullyFilledLeafHandle = fullyFilledNodes.read(this->leaf_level(), firstMaterial);
            const auto fullyFilledLeaf = get_leaf_ptr(fullyFilledLeafHandle);
            constexpr uint32_t leafSizeInU32 = 2 + (64 * Super::MaterialBits + 31) / 32;
            for (uint32_t i = 2; i < leafSizeInU32; ++i) {
                if (fullyFilledLeaf[i] != pLeaf[i])
                    return false;
            }
            outMaterial = firstMaterial;
        } else {
            outMaterial = 0;
        }
        return true;
    }

    HOST_DEVICE uint32_t find_leaf(uint32_t const* pLeaf) const
    {
        const uint32_t leafSizeInU32 = this->get_leaf_size(pLeaf);
        check(leafSizeInU32 >= this->minLeafSizeInU32);
        check(leafSizeInU32 <= this->maxLeafSizeInU32);
        const uint32_t hashTableIndex = leafSizeInU32 - this->minItemSizeInU32;

        auto& hashTable = hashTables[hashTableIndex];
        const uint32_t nodeIdx = hashTable.find(pLeaf);
        if (nodeIdx == HashTableImpl::not_found)
            return Super::invalid_handle;
        else
            return MyGPUHashDAG::create_leaf_handle(nodeIdx, leafSizeInU32);
    }
    DEVICE auto find_leaf_as_warp(uint32_t const* pLeaf) const
    {
        if constexpr (has_add_as_warp<HashTableImpl>) {
            const uint32_t leafSizeInU32 = this->get_leaf_size(pLeaf);
            check(leafSizeInU32 >= this->minLeafSizeInU32);
            check(leafSizeInU32 <= this->maxLeafSizeInU32);
            const uint32_t hashTableIndex = leafSizeInU32 - this->minItemSizeInU32;
            auto& hashTable = hashTables[hashTableIndex];

            const uint32_t nodeIdx = hashTable.findAsWarp(pLeaf);
            if (nodeIdx == HashTableImpl::not_found)
                return Super::invalid_handle;
            else
                return MyGPUHashDAG::create_leaf_handle(nodeIdx, leafSizeInU32);
        }
    }
    HOST_DEVICE uint32_t add_leaf(uint32_t const* pLeaf)
    {
        const uint32_t leafSizeInU32 = this->get_leaf_size(pLeaf);
        check(leafSizeInU32 >= this->minLeafSizeInU32);
        check(leafSizeInU32 <= this->maxLeafSizeInU32);
        const uint32_t hashTableIndex = leafSizeInU32 - this->minItemSizeInU32;
        auto& hashTable = hashTables[hashTableIndex];
        check(hashTable.itemSizeInU32 == leafSizeInU32);

        const uint32_t leafIdx = hashTable.add(pLeaf);
        auto out = MyGPUHashDAG::create_leaf_handle(leafIdx, leafSizeInU32);
        check(MyGPUHashDAG::get_leaf_index(out) == leafIdx);
        check(MyGPUHashDAG::get_leaf_size_from_handle(out) == leafSizeInU32);
#if ENABLE_CHECKS
#ifdef __CUDA_ARCH__
        __threadfence();
#endif
        // const uint32_t* pOut = hashTable.decodePointer(leafIdx);
        // for (uint32_t i = 0; i < leafSizeInU32; ++i) {
        //     check(pOut[i] == pLeaf[i]);
        // }
#endif
        return out;
    }
    DEVICE auto add_leaf_as_warp(uint32_t const* pLeaf)
    {
        if constexpr (has_add_as_warp<HashTableImpl>) {
            const uint32_t leafSizeInU32 = this->get_leaf_size(pLeaf);
            check(leafSizeInU32 >= this->minLeafSizeInU32);
            check(leafSizeInU32 <= this->maxLeafSizeInU32);
            const uint32_t hashTableIndex = leafSizeInU32 - this->minItemSizeInU32;
            auto& hashTable = hashTables[hashTableIndex];
            const uint32_t leafIdx = hashTable.addAsWarp(pLeaf);
            return MyGPUHashDAG::create_leaf_handle(leafIdx, leafSizeInU32);
        }
    }

    // Finds nodes with the same children. This function ignores all header bits except the child mask.
    HOST_DEVICE uint32_t find_node(const uint32_t* pNode) const
    {
        const auto nodeSizeInU32 = MyGPUHashDAG::get_node_size(pNode);
        check(nodeSizeInU32 <= maxNodeSizeInU32);
        const uint32_t hashTableIndex = nodeSizeInU32 - this->minItemSizeInU32;
        const auto& hashTable = hashTables[hashTableIndex];
        const uint32_t nodeIdx = hashTable.find(pNode);

        if (nodeIdx == HashTableImpl::not_found)
            return Super::invalid_handle;
        else
            return MyGPUHashDAG::create_node_handle(nodeIdx, nodeSizeInU32);
    }
    DEVICE uint32_t find_node_as_warp(const uint32_t* pNode) const
    {
        if constexpr (has_add_as_warp<HashTableImpl>) {
            const auto nodeSizeInU32 = MyGPUHashDAG::get_node_size(pNode);
            const uint32_t hashTableIndex = nodeSizeInU32 - this->minItemSizeInU32;
            const auto& hashTable = hashTables[hashTableIndex];
            const uint32_t nodeIdx = hashTable.findAsWarp(pNode);

            if (nodeIdx == HashTableImpl::not_found)
                return Super::invalid_handle;
            else
                return MyGPUHashDAG::create_node_handle(nodeIdx, nodeSizeInU32);
        }
    }
    HOST_DEVICE uint32_t add_node(const uint32_t* pNode)
    {
        const auto nodeSizeInU32 = MyGPUHashDAG::get_node_size(pNode);
        check(nodeSizeInU32 >= this->minNodeSizeInU32);
        // check(nodeSizeInU32 - this->minNodeSizeInU32 < nodeHashTables.size());
        const uint32_t hashTableIndex = nodeSizeInU32 - this->minItemSizeInU32;
        auto& hashTable = hashTables[hashTableIndex];
        check(hashTable.itemSizeInU32 == nodeSizeInU32);
        const uint32_t nodeIdx = hashTable.add(pNode);
        return MyGPUHashDAG::create_node_handle(nodeIdx, nodeSizeInU32);
    }
    DEVICE auto add_node_as_warp(uint32_t const* pNode)
    {
        if constexpr (has_add_as_warp<HashTableImpl>) {
            const auto nodeSizeInU32 = MyGPUHashDAG::get_node_size(pNode);
            const uint32_t hashTableIndex = nodeSizeInU32 - this->minItemSizeInU32;
            auto& hashTable = hashTables[hashTableIndex];
            const uint32_t nodeIdx = hashTable.addAsWarp(pNode);
            return MyGPUHashDAG::create_node_handle(nodeIdx, nodeSizeInU32);
        }
    }

    bool is_valid() const
    {
        return hashTables.is_valid();
    }

    HOST_DEVICE uint32_t get_first_node_index() const
    {
        return firstNodeIndex;
    }

    HOST_DEVICE const LeafDecoder get_leaf_ptr(uint32_t handle) const
    {
        const uint32_t leafSizeInU32 = MyGPUHashDAG::get_leaf_size_from_handle(handle);
        const uint32_t hashTableIndex = leafSizeInU32 - this->minItemSizeInU32;
        const uint32_t indexInHashTable = MyGPUHashDAG::get_leaf_index(handle);
        return hashTables[hashTableIndex].decodePointer(indexInHashTable);
    }
    HOST_DEVICE const NodeDecoder get_node_ptr(uint32_t level, uint32_t handle) const
    {
        check(level < this->leaf_level());
        const uint32_t nodeSizeInU32 = MyGPUHashDAG::get_node_size_from_handle(handle);
        const uint32_t hashTableIndex = nodeSizeInU32 - this->minItemSizeInU32;
        const uint32_t indexInHashTable = MyGPUHashDAG::get_node_index(handle);
        return hashTables[hashTableIndex].decodePointer(indexInHashTable);
    }

    HOST_DEVICE void markNodeAsActive(uint32_t handle)
    {
        const uint32_t nodeSizeInU32 = MyGPUHashDAG::get_node_size_from_handle(handle);
        const uint32_t hashTableIndex = nodeSizeInU32 - this->minItemSizeInU32;
        const uint32_t indexInHashTable = MyGPUHashDAG::get_node_index(handle);
        hashTables[hashTableIndex].markAsActive(indexInHashTable);
    }
    HOST_DEVICE void markLeafAsActive(uint32_t handle)
    {
        const uint32_t nodeSizeInU32 = MyGPUHashDAG::get_leaf_size_from_handle(handle);
        const uint32_t hashTableIndex = nodeSizeInU32 - this->minItemSizeInU32;
        const uint32_t indexInHashTable = MyGPUHashDAG::get_leaf_index(handle);
        hashTables[hashTableIndex].markAsActive(indexInHashTable);
    }

    void reserveIfNecessary(gsl::span<const uint32_t> numItems);
    void garbageCollect(gsl::span<const uint32_t> activeRootNodes);

    void report(StatsRecorder&) const;

private:
    template <EMemoryType>
    friend struct MyGPUHashDAG;

    // StaticArray instead of std::array because:
    // A) struct becomes too large to pass to CUDA (running out of "formal parameter space")
    // B) allow to share hash tables between different DAGs (e.g. geometry & material DAGs)
    StaticArray<HashTableImpl> hashTables;
};
