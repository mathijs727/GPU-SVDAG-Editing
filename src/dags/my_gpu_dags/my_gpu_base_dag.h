#pragma once
#include "configuration/gpu_hash_dag_definitions.h"
#include "dags/base_dag.h"
#include "dags/hash_dag/hash_dag_globals.h"
#include "typedefs.h"
#include <concepts>

// clang-format off
template <typename T>
concept ElementEncoder = requires(const T& e, int i)
{
    { e[i] } -> std::convertible_to<uint32_t>;
};
// clang-format on

class MyGPUHashDAGUndoRedo {
public:
    struct Frame {
        uint32 firstNodeIndex = 0;
    };

public:
    void add_frame(const Frame& frame)
    {
        PROFILE_FUNCTION();

        // Due to the cursor being behind by 1 or 2 frames (to prevent cudaDeviceSynchronize each frame) there may
        // sometimes be duplicate edits. If these result in the exact same tree structure then we don't store the frame.
        if (!m_undoFrames.empty() && m_undoFrames.back().firstNodeIndex == frame.firstNodeIndex)
            return;

        m_redoFrames.clear();
        m_undoFrames.push_back(frame);
    }
    template <typename TDAG>
    void undo(TDAG& dag)
    {
        if (m_undoFrames.empty())
            return;

        const Frame undoFrame = m_undoFrames.back();
        m_undoFrames.pop_back();

        Frame redoFrame;
        redoFrame.firstNodeIndex = dag.firstNodeIndex;
        m_redoFrames.push_back(redoFrame);

        dag.firstNodeIndex = undoFrame.firstNodeIndex;
    }
    template <typename TDAG>
    void redo(TDAG& dag)
    {
        if (m_redoFrames.empty())
            return;

        const Frame redoFrame = m_redoFrames.back();
        m_redoFrames.pop_back();

        Frame undoFrame;
        undoFrame.firstNodeIndex = dag.firstNodeIndex;
        m_undoFrames.push_back(undoFrame);

        dag.firstNodeIndex = redoFrame.firstNodeIndex;
    }

    template <typename TDAG>
    void garbageCollect(TDAG& dag)
    {
        // if (m_undoFrames.size() + m_redoFrames.size() == 0)
        //     return;
        m_undoFrames.clear();
        m_redoFrames.clear();

        printf("Freeing unused subtrees...\n");
        std::array activeRootNodes { dag.get_first_node_index() };
        dag.garbageCollect(activeRootNodes);
    }

private:
    std::vector<Frame> m_undoFrames;
    std::vector<Frame> m_redoFrames;
};

static constexpr uint32_t SentinelMaterial = 0xFFFFFFFF;

template <typename TChild, uint32_t MaterialBits_>
struct GPUBaseDAG : public BaseDAG {
public:
    static constexpr uint32_t MaterialBits = MaterialBits_;
    static constexpr uint32_t NumMaterials = 1u << MaterialBits;

    // Header size is always 1. Ranges from 1+1=2 to 1+8=9 (1 to 8 children per node).
    static constexpr uint32_t minNodeSizeInU32 = 2;
    static constexpr uint32_t maxNodeSizeInU32 = 9;
    static constexpr uint32_t minLeafSizeInU32 = 2 + (MaterialBits ? 1 : 0);
    static constexpr uint32_t maxLeafSizeInU32 = 2 + (64 * MaterialBits + 31) / 32;
    static constexpr uint32_t minItemSizeInU32 = std::min(minNodeSizeInU32, minLeafSizeInU32);
    static constexpr uint32_t maxItemSizeInU32 = std::max(maxNodeSizeInU32, maxLeafSizeInU32);

    static constexpr uint32_t invalid_handle = 0xFFFFFFFF;

    class LeafBuilder_ {
    public:
        constexpr HOST_DEVICE LeafBuilder_(uint32_t* pLeaf)
            : pLeaf(pLeaf)
        {
            for (uint32_t i = 0; i < maxLeafSizeInU32; ++i)
                pLeaf[i] = 0;
        }

        constexpr HOST_DEVICE void set(uint32_t value)
        {
            if constexpr (MaterialBits == 0) {
                // Update bitmask.
                pLeaf[voxelIndex >> 5] |= 1u << (voxelIndex & 31);
            } else {
                check(value < NumMaterials);

                // Update bitmask.
                pLeaf[voxelIndex >> 5] |= 1u << (voxelIndex & 31);

                // Store material.
                const uint32_t voxelOffsetInBits = voxelOffset * MaterialBits;
                const uint32_t voxelOffsetInU32 = voxelOffsetInBits >> 5;
                uint32_t voxelOffsetWithinU32 = voxelOffsetInBits & 31;
                pLeaf[2 + voxelOffsetInU32] |= value << voxelOffsetWithinU32;
                if constexpr (32 % MaterialBits != 0) {
                    if (voxelOffsetWithinU32 + MaterialBits > 32) {
                        value >>= 32 - voxelOffsetWithinU32;
                        pLeaf[3 + voxelOffsetInU32] = value;
                    }
                }

                ++voxelOffset;
            }
        }
        constexpr HOST_DEVICE void next() { ++voxelIndex; }
        constexpr HOST_DEVICE uint32_t finalize() const
        {
            check(voxelIndex == 64);
            if constexpr (MaterialBits == 0)
                return 2;
            else
                return 2 + (voxelOffset * MaterialBits + 31) / 32;
        }

    private:
        uint32_t* pLeaf;
        uint32_t voxelIndex = 0, voxelOffset = 0;
    };

    static HOST_DEVICE uint32_t* encode_node_header(uint32_t* pNode, uint32_t childMask)
    {
        uint32_t header = 0;
        Utils::insert_bits<headerBitmask_offset, headerBitmask_size>(header, childMask);
        *pNode = header;
        return pNode + get_header_size();
    }
    static HOST_DEVICE uint32_t* encode_node_header_with_payload(uint32_t* pNode, uint32_t childMask, uint32_t userPayload)
    {
        uint32_t header = 0;
        Utils::insert_bits<headerBitmask_offset, headerBitmask_size>(header, childMask);
        Utils::insert_bits<headerUserPayload_offset, headerUserPayload_size>(header, userPayload);
        *pNode = header;
        return pNode + get_header_size();
    }
    static HOST_DEVICE uint32_t* encode_node_header(uint32_t* pNode, uint32_t childMask, uint32_t fullyFilledMaterial)
    {
        uint32_t header = 0;
        Utils::insert_bits<headerBitmask_offset, headerBitmask_size>(header, childMask);
        Utils::insert_bits<headerHasFullyFilledMaterial_offset, headerHasFullyFilledMaterial_size>(header, 1u);
        Utils::insert_bits<headerFullyFilledMaterial_offset, headerFullyFilledMaterial_size>(header, fullyFilledMaterial);
        *pNode = header;
        return pNode + get_header_size();
    }
    HOST_DEVICE static uint32_t get_node_child_mask(uint32_t header)
    {
        // WARNING: must match Utils::child_mask(header) or some code (e.g. rendering) will break.
        return Utils::extract_bits<headerBitmask_offset, headerBitmask_size>(header);
    }
    HOST_DEVICE static uint32_t get_node_user_payload(uint32_t header)
    {
        return Utils::extract_bits<headerUserPayload_offset, headerUserPayload_size>(header);
    }
    HOST_DEVICE static bool get_node_is_fully_filled(uint32_t header, uint32_t& outMaterial)
    {
        outMaterial = Utils::extract_bits<headerFullyFilledMaterial_offset, headerFullyFilledMaterial_size>(header);
        return Utils::extract_bits<headerHasFullyFilledMaterial_offset, headerHasFullyFilledMaterial_size>(header);
    }

    HOST_DEVICE static constexpr uint32_t get_header_size()
    {
        return 1;
    }
    HOST_DEVICE static uint32_t get_leaf_size(const ElementEncoder auto pLeaf)
    {
        const auto voxelCount = Utils::popc(pLeaf[0]) + Utils::popc(pLeaf[1]);
        return 2 + ((voxelCount * MaterialBits + 31) >> 5); // Divide by 32.
    }
    HOST_DEVICE static constexpr uint32_t get_node_size(const ElementEncoder auto pNode)
    {
        return get_header_size() + Utils::popc(Utils::child_mask(pNode[0]));
    }

    HOST_DEVICE static uint32 child_offset(uint8 childMask, uint8 child)
    {
        return Utils::popc(childMask & ((1u << child) - 1u));
    }

    HOST_DEVICE static uint32_t create_leaf_handle(uint32_t index, uint32_t leafSizeInU32)
    {
        uint32_t handle = 0;
        Utils::insert_bits<leafPointerIndex_offset, leafPointerIndex_size>(handle, index);
        Utils::insert_bits<leafPointerSize_offset, leafPointerSize_size>(handle, leafSizeInU32 - minLeafSizeInU32);
        check(get_leaf_size_from_handle(handle) == leafSizeInU32);
        check(get_leaf_index(handle) == index);
        return handle;
    }
    HOST_DEVICE static constexpr uint32_t get_leaf_size_from_handle(uint32_t handle)
    {
        if constexpr (leafPointerSize_size == 0)
            return minLeafSizeInU32;
        else
            return minLeafSizeInU32 + Utils::extract_bits<leafPointerSize_offset, leafPointerSize_size>(handle);
    }

    HOST_DEVICE static uint32_t create_node_handle(uint32_t index, uint32_t nodeSizeInU32)
    {
        uint32_t handle = 0;
        Utils::insert_bits<nodePointerIndex_offset, nodePointerIndex_size>(handle, index);
        Utils::insert_bits<nodePointerSize_offset, nodePointerSize_size>(handle, nodeSizeInU32 - minNodeSizeInU32);
        check(get_node_size_from_handle(handle) == nodeSizeInU32);
        check(get_node_index(handle) == index);
        return handle;
    }
    HOST_DEVICE static constexpr uint32_t get_node_size_from_handle(uint32_t handle)
    {
        return minNodeSizeInU32 + Utils::extract_bits<nodePointerSize_offset, nodePointerSize_size>(handle);
    }

    HOST_DEVICE uint32_t get_node(uint32_t level, uint32_t handle) const
    {
        checkInf(level, leaf_level());
        return self().get_node_ptr(level, handle)[0];
    }
    HOST_DEVICE uint32_t get_child_index(uint32_t level, uint32_t handle, uint8_t childMask, uint8_t child) const
    {
        const auto pNode = self().get_node_ptr(level, handle);
        return pNode[get_header_size() + child_offset(childMask, child)];
    }
    HOST_DEVICE uint32_t get_child_index(uint32_t level, const ElementEncoder auto pNode, uint8_t childMask, uint8_t child) const
    {
        return pNode[get_header_size() + child_offset(childMask, child)];
    }
    HOST_DEVICE uint32_t get_child_index(uint32_t level, uint32_t handle, uint8_t childOffset) const
    {
        const auto pNode = self().get_node_ptr(level, handle);
        return pNode[get_header_size() + childOffset];
    }

    HOST_DEVICE bool is_leaf_fully_filled(const ElementEncoder auto pLeaf, uint32_t& outMaterial) const
    {
        if (pLeaf[0] != 0xFFFFFFFF || pLeaf[1] != 0xFFFFFFFF)
            return false;

        if constexpr (MaterialBits > 0) {
            static_assert(MaterialBits < 32);
            constexpr uint32_t MaterialMask = (1u << MaterialBits) - 1u;
            const uint32_t firstMaterial = pLeaf[2] & MaterialMask;
            for (uint32_t i = 0; i < 64; ++i) {
                uint32_t material = (uint32_t)-1;
                get_material(pLeaf, i, material);
                if (material != firstMaterial)
                    return false;
            }
            outMaterial = firstMaterial;
        } else {
            outMaterial = 0;
        }

        return true;
    }

    // Gets the uint64_t bitmask that represents a 4x4x4 region.
    HOST_DEVICE Leaf get_leaf(uint32_t handle) const
    {
        return get_leaf(self().get_leaf_ptr(handle));
    }
    HOST_DEVICE Leaf get_leaf(const ElementEncoder auto pLeaf) const
    {
        return { pLeaf[0], pLeaf[1] };
    }
    HOST_DEVICE static bool get_material(const ElementEncoder auto pLeaf, uint32_t voxelIndex, uint32_t& outMaterial)
    {
        const uint64_t bitmask = (uint64_t)pLeaf[0] | ((uint64_t)pLeaf[1] << 32);
        if (!(bitmask & (1llu << voxelIndex)))
            return false;

        if constexpr (MaterialBits == 0) {
            outMaterial = 0;
        } else {
            static constexpr uint32_t MaterialMask = NumMaterials - 1u;
            const uint32_t voxelOffsetInBits = Utils::popcll(bitmask & ((1llu << voxelIndex) - 1llu)) * MaterialBits;
            const uint32_t voxelOffsetInU32 = voxelOffsetInBits >> 5; // same as division by 32.
            const uint32_t voxelOffsetWithinU32 = voxelOffsetInBits & 31;
            outMaterial = (pLeaf[2 + voxelOffsetInU32] >> voxelOffsetWithinU32) & MaterialMask;

            if constexpr (32 % MaterialBits != 0) {
                if (voxelOffsetWithinU32 + MaterialBits > 32) {
                    const uint32_t numOverflowBits = 32 - voxelOffsetWithinU32;
                    outMaterial |= ((pLeaf[3 + voxelOffsetInU32] << numOverflowBits) & MaterialMask);
                }
            }
        }
        return true;
    }

protected:
    HOST_DEVICE static uint32_t get_leaf_index(uint32_t handle)
    {
        return Utils::extract_bits<leafPointerIndex_offset, leafPointerIndex_size>(handle);
    }
    HOST_DEVICE static uint32_t get_node_index(uint32_t handle)
    {
        return Utils::extract_bits<nodePointerIndex_offset, nodePointerIndex_size>(handle);
    }

private:
    HOST_DEVICE const TChild& self() const
    {
        return static_cast<const TChild&>(*this);
    }

    static constexpr uint32_t nodePointerSize_size = std::bit_width(maxNodeSizeInU32 - minNodeSizeInU32);
    static constexpr uint32_t nodePointerSize_offset = 32 - nodePointerSize_size;
    static constexpr uint32_t nodePointerIndex_size = 32 - nodePointerSize_size;
    static constexpr uint32_t nodePointerIndex_offset = 0;
    
    static constexpr uint32_t leafPointerSize_size = std::bit_width(maxLeafSizeInU32 - minLeafSizeInU32);
    static constexpr uint32_t leafPointerSize_offset = 32 - leafPointerSize_size;
    static constexpr uint32_t leafPointerIndex_size = 32 - leafPointerSize_size;
    static constexpr uint32_t leafPointerIndex_offset = 0;

    static constexpr uint32_t headerBitmask_offset = 0;
    static constexpr uint32_t headerBitmask_size = 8;
    static constexpr uint32_t headerHasFullyFilledMaterial_offset = headerBitmask_offset + headerBitmask_size;
    static constexpr uint32_t headerHasFullyFilledMaterial_size = 1;
    static constexpr uint32_t headerFullyFilledMaterial_offset = headerHasFullyFilledMaterial_offset + headerHasFullyFilledMaterial_size;
    static constexpr uint32_t headerFullyFilledMaterial_size = MaterialBits;
    static constexpr uint32_t headerUserPayload_offset = headerFullyFilledMaterial_offset + headerFullyFilledMaterial_size;
    static constexpr uint32_t headerUserPayload_size = 32 - headerUserPayload_offset;
};