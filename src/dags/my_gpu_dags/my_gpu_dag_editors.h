#pragma once
#include "dags/dag_utils.h"
#include "path.h"
#include "typedefs.h"
#include <cuda_math.h>

enum class NodeState {
    Partial,
    Filled,
    Empty
};
struct NodeDesc {
    NodeState state;
    uint32_t fillMaterial;
};

enum class EditOperation {
    Subdivide, // Want to edit; subdivide further
    Fill, // Want to completely fill
    Empty, // Want to completely empty
    Keep // Keep current node without modifications.
};
struct NodeEditDesc {
    EditOperation operation;
    uint32_t fillMaterial;
    float3 fillColor;
};

template <typename TChild>
struct MyGpuEditor {
    constexpr static bool is_gpu_editor = true;

    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32 depth, const NodeDesc& nodeDesc) const
    {
        const float3 start = path.as_position(depth);
        return self().should_edit_impl(start, start + make_float3(float(1u << depth)), nodeDesc);
    }

private:
    HOST_DEVICE const TChild& self() const
    {
        return static_cast<const TChild&>(*this);
    }
};
struct MyGpuBoxEditorBase {
    float radius;
    const float3 boundsMin;
    const float3 boundsMax;

    MyGpuBoxEditorBase(const float3& center, float radius)
        : radius(radius)
        , boundsMin(center - make_float3(radius))
        , boundsMax(radius == 0 ? center + make_float3(1.0f) : center + make_float3(radius))
    {
    }

    HOST_DEVICE bool test_node_overlap(const Path& path, const uint32_t depth) const
    {
        const float3 start = path.as_position(depth);
        const float3 nodeMin = start;
        const float3 nodeMax = start + make_float3(float(1u << depth));
        return test_node_overlap(nodeMin, nodeMax);
    }
    HOST_DEVICE bool test_node_overlap(const float3& nodeMin, const float3& nodeMax) const
    {
        return !(boundsMin.x >= nodeMax.x || boundsMin.y >= nodeMax.y || boundsMin.z >= nodeMax.z || boundsMax.x <= nodeMin.x || boundsMax.y <= nodeMin.y || boundsMax.z <= nodeMin.z || nodeMin.x >= boundsMax.x || nodeMin.y >= boundsMax.y || nodeMin.z >= boundsMax.z || nodeMax.x <= boundsMin.x || nodeMax.y <= boundsMin.y || nodeMax.z <= boundsMin.z);
    }

    HOST_DEVICE bool test_node_fully_contained(const Path& path, uint32_t depth) const
    {
        const float3 start = path.as_position(depth);
        const float3 nodeMin = start;
        const float3 nodeMax = start + make_float3(float(1u << depth));
        return test_node_fully_contained(nodeMin, nodeMax);
    }
    HOST_DEVICE bool test_node_fully_contained(const float3& nodeMin, const float3& nodeMax) const
    {
        return boundsMin.x <= nodeMin.x && boundsMin.y <= nodeMin.y && boundsMin.z <= nodeMin.z && boundsMax.x >= nodeMax.x && boundsMax.y >= nodeMax.y && boundsMax.z >= nodeMax.z;
    }
};

template <bool isAdding>
struct MyGpuBoxEditor : public MyGpuBoxEditorBase, MyGpuEditor<MyGpuBoxEditor<isAdding>> {
    using MyGpuBoxEditorBase::MyGpuBoxEditorBase;

    uint32_t fillMaterial;

    MyGpuBoxEditor(float3 center, float radius, uint32_t material)
        : MyGpuBoxEditorBase(center, radius)
        , fillMaterial(material)
    {
    }

    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32_t depth, NodeDesc nodeDesc) const
    {
        const float3 start = path.as_position(depth);
        const float3 nodeMin = start;
        const float3 nodeMax = start + make_float3(float(1u << depth));

        NodeEditDesc out;
        if (test_node_fully_contained(nodeMin, nodeMax)) {
            out.operation = isAdding ? EditOperation::Fill : EditOperation::Empty;
            out.fillMaterial = fillMaterial;
        } else if (test_node_overlap(nodeMin, nodeMax)) {
            out.operation = EditOperation::Subdivide;
        } else {
            out.operation = EditOperation::Keep;
        }
        return out;
    }

    HOST_DEVICE std::optional<uint32_t> get_new_value(float3 position, std::optional<uint32_t>) const
    {
        if (isAdding)
            return fillMaterial;
        else
            return {};
    }
};

static HOST_DEVICE bool aabbOverlap(const float3& lhsMin, const float3& lhsMax, const float3& rhsMin, const float3& rhsMax)
{
    return !(
        lhsMin.x >= rhsMax.x || lhsMin.y >= rhsMax.y || lhsMin.z >= rhsMax.z || lhsMax.x <= rhsMin.x || lhsMax.y <= rhsMin.y || lhsMax.z <= rhsMin.z || rhsMin.x >= lhsMax.x || rhsMin.y >= lhsMax.y || rhsMin.z >= lhsMax.z || rhsMax.x <= lhsMin.x || rhsMax.y <= lhsMin.y || rhsMax.z <= lhsMin.z);
}

#if 1
// Queries every voxel individually
template <typename TDag>
struct MyGpuCopyEditor final : MyGpuBoxEditorBase, MyGpuEditor<MyGpuCopyEditor<TDag>> {
    // Copy of the dag as we want to access the old one
    const TDag dagCopy;
    // But ref of the colors in case the arrays are reallocated cuz too small
    // const float3 source;
    // const float3 dest;
    const float3 fDestToSource;
    const int3 iDestToSource;

    struct State {
        struct Parent {
            Path path;
            uint32_t handle;
        };
        Parent parents[8];
    };
    static constexpr uint32_t minStateFullDepth = 2;

    State createInitialState() const
    {
        State out {};
        out.parents[0].path.path = make_uint3(0);
        out.parents[0].handle = dagCopy.get_first_node_index();
        out.parents[1].handle = 0xFFFFFFFF;
        return out;
    }

    MyGpuCopyEditor(const TDag& dagCopy, const float3& source, const float3& dest, const float3& center, float radius)

        : MyGpuBoxEditorBase(dest + (center - source), radius)
        , dagCopy(dagCopy)
        , fDestToSource(-dest + source)
        , iDestToSource(make_int3((int)fDestToSource.x, (int)fDestToSource.y, (int)fDestToSource.z))
    {
    }

    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32 depth, const NodeDesc& nodeDesc) const
    {
        if (test_node_overlap(path, depth))
            return { EditOperation::Subdivide };
        else
            return { EditOperation::Keep };
    }
    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32 depth, const NodeDesc& nodeDesc, const State& inState, State& outState) const
    {
        const float3 nodeMin = path.as_position(depth);
        const auto nodeSize = float(1u << depth);
        const float3 nodeMax = nodeMin + make_float3(nodeSize);

        if (!test_node_overlap(nodeMin, nodeMax))
            return { EditOperation::Keep };

        const float3 shiftedNodeMin = max(nodeMin, boundsMin) + fDestToSource;
        const float3 shiftedNodeMax = min(shiftedNodeMin + make_float3(nodeSize), boundsMax + fDestToSource);

        const uint32_t parentLevel = MAX_LEVELS - depth - 1;
        int numOutParents = 0;
        for (auto& parent : inState.parents) {
            if (parent.handle == 0xFFFFFFFF)
                break;

            const auto pNode = dagCopy.get_node_ptr(parentLevel, parent.handle);
            const uint8 childMask = Utils::child_mask(pNode[0]);
            for (uint8_t childIdx = 0; childIdx < 8; ++childIdx) {
                if (!(childMask & (1u << childIdx)))
                    continue;

                Path childPath = parent.path;
                childPath.descend(childIdx);
                const auto childMin = childPath.as_position(depth);
                const auto childMax = childMin + make_float3(nodeSize);

                if (aabbOverlap(childMin, childMax, shiftedNodeMin, shiftedNodeMax)) {
                    outState.parents[numOutParents].handle = dagCopy.get_child_index(parentLevel, pNode, childMask, childIdx);
                    check(numOutParents < 8);
                    outState.parents[numOutParents].path = childPath;
                    ++numOutParents;
                }
            }
        }
        if (numOutParents != 8)
            outState.parents[numOutParents].handle = 0xFFFFFFFF;

        if (numOutParents == 0)
            return { EditOperation::Keep };
        else
            return { EditOperation::Subdivide };
    }

    /*HOST_DEVICE std::optional<uint32_t> get_new_value(const float3& position, std::optional<uint32_t> oldValue) const
    {
        const float3 offset = position + fDestToSource;
        const Path path { (unsigned)offset.x, (unsigned)offset.y, (unsigned)offset.z };
        return DAGUtils::get_value(dagCopy, path);
    }*/
    HOST_DEVICE std::optional<uint32_t> get_new_value(const uint3& destPosition, std::optional<uint32_t> oldValue, const State& upperParentState) const
    {
        //const float3 offset = make_float3(destPosition) + fDestToSource;
        //const Path path { (unsigned)offset.x, (unsigned)offset.y, (unsigned)offset.z };
        //return DAGUtils::get_value(dagCopy, path);

        uint3 sourcePosition = make_uint3(max(make_int3(destPosition) + iDestToSource, make_int3(0, 0, 0)));
        uint3 parentPath = sourcePosition;
        parentPath.x >>= 2;
        parentPath.y >>= 2;
        parentPath.z >>= 2;
        typename State::Parent const* pParent = nullptr;
        for (const auto& parent : upperParentState.parents) {
            if (parent.handle == 0xFFFFFFFF)
                break;
            if (parent.path == parentPath)
                pParent = &parent;
        }
        if (!pParent)
            return oldValue;
            
        const uint8 leafBitIndex = uint8(
            (((sourcePosition.x & 0x1) == 0) ? 0 : 4) | (((sourcePosition.y & 0x1) == 0) ? 0 : 2) | (((sourcePosition.z & 0x1) == 0) ? 0 : 1) | (((sourcePosition.x & 0x2) == 0) ? 0 : 32) | (((sourcePosition.y & 0x2) == 0) ? 0 : 16) | (((sourcePosition.z & 0x2) == 0) ? 0 : 8));
        uint32_t outMaterial;
        if (dagCopy.get_material(dagCopy.get_leaf_ptr(pParent->handle), leafBitIndex, outMaterial))
            oldValue = outMaterial;
        return oldValue;
    }
};

#else

// Queries every voxel individually
template <typename TDag>
struct MyGpuCopyEditor final : MyGpuBoxEditorBase, MyGpuEditor<MyGpuCopyEditor<TDag>> {
    // Copy of the dag as we want to access the old one
    const TDag dagCopy;
    // But ref of the colors in case the arrays are reallocated cuz too small
    // const float3 source;
    // const float3 dest;
    const float3 fDestToSource;
    const uint3 uDestToSource;

#define STATE State

    struct STATE {
        struct Parent {
            Path path;
            uint32_t handle;
        };
        Parent parents[8];
    };
    static constexpr uint32_t minStateFullDepth = 2;

    STATE createInitialState() const
    {
        STATE out {};
        out.parents[0].path.path = make_uint3(0);
        out.parents[0].handle = dagCopy.get_first_node_index();
        out.parents[1].handle = 0xFFFFFFFF;
        return out;
    }

    MyGpuCopyEditor(const TDag& dagCopy, const float3& source, const float3& dest, const float3& center, float radius)

        : MyGpuBoxEditorBase(dest + (center - source), radius)
        , dagCopy(dagCopy)
        , fDestToSource(-dest + source)
        , uDestToSource(make_uint3((unsigned)fDestToSource.x, (unsigned)fDestToSource.y, (unsigned)fDestToSource.z))
    {
    }

    HOST_DEVICE NodeEditDesc should_edit_impl(const float3& nodeMin, const float3& nodeMax, const NodeDesc& nodeDesc) const
    {
        if (test_node_overlap(nodeMin, nodeMax))
            return { EditOperation::Subdivide };
        else
            return { EditOperation::Keep };
    }
    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32 depth, const NodeDesc& nodeDesc) const
    {
        const float3 start = path.as_position(depth);
        return should_edit_impl(start, start + make_float3(float(1u << depth)), nodeDesc);
    }

    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32 depth, const NodeDesc& nodeDesc, const STATE& inState, STATE& outState) const
    {
        const float3 nodeMin = path.as_position(depth);
        const auto nodeSize = float(1u << depth);
        const float3 nodeMax = nodeMin + make_float3(nodeSize);

        if (!test_node_overlap(nodeMin, nodeMax))
            return { EditOperation::Keep };

        const float3 shiftedNodeMin = max(nodeMin, boundsMin) + fDestToSource;
        const float3 shiftedNodeMax = min(shiftedNodeMin + make_float3(nodeSize), boundsMax + fDestToSource);

        const uint32_t parentLevel = MAX_LEVELS - depth - 1;
        int numOutParents = 0;
        for (auto& parent : inState.parents) {
            if (parent.handle == 0xFFFFFFFF)
                break;

            const auto pNode = dagCopy.get_node_ptr(parentLevel, parent.handle);
            const uint8 childMask = Utils::child_mask(pNode[0]);
            for (uint8_t childIdx = 0; childIdx < 8; ++childIdx) {
                if (!(childMask & (1u << childIdx)))
                    continue;

                Path childPath = parent.path;
                childPath.descend(childIdx);
                const auto childMin = childPath.as_position(depth);
                const auto childMax = childMin + make_float3(nodeSize);

                if (aabbOverlap(childMin, childMax, shiftedNodeMin, shiftedNodeMax)) {
                    outState.parents[numOutParents].handle = dagCopy.get_child_index(parentLevel, pNode, childMask, childIdx);
                    outState.parents[numOutParents].path = childPath;
                    ++numOutParents;
                }
            }
        }
        if (numOutParents != 8)
            outState.parents[numOutParents].handle = 0xFFFFFFFF;

        if (numOutParents == 0)
            return { EditOperation::Keep };
        else
            return { EditOperation::Subdivide };
    }

    HOST_DEVICE std::optional<uint32_t> get_new_value(const uint3& destPosition, std::optional<uint32_t> oldValue, const STATE& inState) const
    {
        uint3 sourcePosition = destPosition + uDestToSource;
        uint3 parentPath = sourcePosition;
        parentPath.x >>= 2;
        parentPath.y >>= 2;
        parentPath.z >>= 2;
        for (auto& parent : inState.parents) {
            if (parent.handle == 0xFFFFFFFF)
                break;

            if (parent.path == parentPath) {
                const uint8 leafBitIndex = uint8(
                    (((sourcePosition.x & 0x1) == 0) ? 0 : 4) | (((sourcePosition.y & 0x1) == 0) ? 0 : 2) | (((sourcePosition.z & 0x1) == 0) ? 0 : 1) | (((sourcePosition.x & 0x2) == 0) ? 0 : 32) | (((sourcePosition.y & 0x2) == 0) ? 0 : 16) | (((sourcePosition.z & 0x2) == 0) ? 0 : 8));
                uint32_t outMaterial;
                if (dagCopy.get_material(dagCopy.get_leaf_ptr(parent.handle), leafBitIndex, outMaterial))
                    return outMaterial;
                else
                    return {};
            }
        }

        return {};
    }
    HOST_DEVICE std::optional<uint32_t> get_new_value(const float3& position, std::optional<uint32_t> oldValue) const
    {
        const float3 offset = position + fDestToSource;
        const Path path { (unsigned)offset.x, (unsigned)offset.y, (unsigned)offset.z };
        return DAGUtils::get_value(dagCopy, path);
    }
    template <typename T>
    HOST_DEVICE CompressedColor get_new_color(const float3& position, const T& oldColor, bool oldValue, bool newValue) const
    {
        if (oldValue == newValue || !newValue) {
            return oldColor();
        } else {
            CompressedColor out;
            out.set_single_color(make_float3(1, 0, 0));
            return out;
        }
    }

#undef STATE
};
#endif

struct MyGpuSphereEditorBase {
    const float3 center;
    const float radius;
    const float radiusSquared;

    MyGpuSphereEditorBase(float3 center, float radius_)
        : center(center + make_float3(0.5f))
        , radius(std::max(radius_, 0.1f))
        , radiusSquared(radius * radius)
    {
    }

    HOST_DEVICE bool test_node_overlap(const Path& path, uint32_t depth) const
    {
        const float3 start = path.as_position(depth);
        const float3 nodeMin = start;
        const float3 nodeMax = start + make_float3(float(1u << depth));
        return test_node_overlap(nodeMin, nodeMax);
    }
    HOST_DEVICE bool test_node_overlap(const float3& nodeMin, const float3& nodeMax) const
    {
        float dist = radiusSquared;

        if (center.x < nodeMin.x)
            dist -= squared(center.x - nodeMin.x);
        else if (center.x > nodeMax.x)
            dist -= squared(center.x - nodeMax.x);

        if (center.y < nodeMin.y)
            dist -= squared(center.y - nodeMin.y);
        else if (center.y > nodeMax.y)
            dist -= squared(center.y - nodeMax.y);

        if (center.z < nodeMin.z)
            dist -= squared(center.z - nodeMin.z);
        else if (center.z > nodeMax.z)
            dist -= squared(center.z - nodeMax.z);

        return dist > 0;
    }

    HOST_DEVICE bool test_node_fully_contained(const float3& nodeMin, const float3& nodeMax) const
    {
        return std::max(
                   squared(nodeMin.x - center.x),
                   squared(nodeMax.x - center.x))
            + std::max(
                squared(nodeMin.y - center.y),
                squared(nodeMax.y - center.y))
            + std::max(
                squared(nodeMin.z - center.z),
                squared(nodeMax.z - center.z))
            < radiusSquared;
    }
};

template <bool isAdding>
struct MyGpuSphereEditor : public MyGpuSphereEditorBase, MyGpuEditor<MyGpuSphereEditor<isAdding>> {
    uint32_t fillMaterial;

    MyGpuSphereEditor(float3 center, float radius, uint32_t material)
        : MyGpuSphereEditorBase(center, radius)
        , fillMaterial(material)
    {
    }

    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32_t depth, NodeDesc nodeDesc) const
    {
        // A voxel is painted of it **overlaps** with the sphere.
        // Thus a region is painted if the sphere touches all pixels; it does not have to fully overlap all voxels!
        const float3 start = path.as_position(depth);
        const float3 nodeMin = start;
        const float3 nodeMax = start + make_float3(float(1u << depth));
        const bool overlaps = test_node_overlap(nodeMin, nodeMax);
        const bool fullyContained = test_node_fully_contained(nodeMin + 1.0f, nodeMax - 1.0f);

        NodeEditDesc out;
        if (!overlaps) {
            out.operation = EditOperation::Keep;
        } else if (fullyContained) {
            out.operation = isAdding ? EditOperation::Fill : EditOperation::Empty;
            out.fillMaterial = fillMaterial;
        } else {
            out.operation = EditOperation::Subdivide;
        }
        return out;
    }

    HOST_DEVICE std::optional<uint32_t> get_new_value(float3 position, std::optional<uint32_t>) const
    {
        if (isAdding)
            return fillMaterial;
        else
            return {};
    }
};

struct MyGpuSpherePaintEditor final : MyGpuSphereEditorBase, MyGpuEditor<MyGpuSpherePaintEditor> {
    float3 paintColor;
    uint32_t paintMaterial;

    MyGpuSpherePaintEditor(float3 center, float radius, float3 paintColor, uint32_t material)
        : MyGpuSphereEditorBase(center, radius)
        , paintColor(paintColor)
        , paintMaterial(material)
    {
    }

    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32_t depth, NodeDesc nodeDesc) const
    {
        const float3 start = path.as_position(depth);
        const float3 nodeMin = start;
        const float3 nodeMax = start + make_float3(float(1u << depth));
        const bool overlaps = test_node_overlap(nodeMin, nodeMax);
        const bool fullyContained = test_node_fully_contained(nodeMin, nodeMax);

        NodeEditDesc out;
        if (!overlaps) {
            out.operation = EditOperation::Keep;
        } else if (fullyContained && nodeDesc.state == NodeState::Filled) {
            out.operation = EditOperation::Fill;
            out.fillMaterial = paintMaterial;
            out.fillColor = paintColor;
        } else if (nodeDesc.state == NodeState::Empty) {
            out.operation = EditOperation::Empty;
        } else {
            out.operation = EditOperation::Subdivide;
        }
        return out;
    }

    HOST_DEVICE std::optional<uint32_t> get_new_value(float3 position, std::optional<uint32_t> oldValue) const
    {
        return paintMaterial;
    }
};

template <typename TDAG>
struct MyGpuSphereErosionEditor final : MyGpuSphereEditorBase, MyGpuEditor<MyGpuSphereErosionEditor<TDAG>> {
    TDAG dag;
    uint32_t threshold;

    MyGpuSphereErosionEditor(TDAG dag, float3 center, float radius, uint32_t threshold)
        : MyGpuSphereEditorBase(center, radius)
        , dag(dag)
        , threshold(threshold)
    {
    }

    HOST_DEVICE NodeEditDesc should_edit(const Path& path, uint32_t depth, NodeDesc nodeDesc) const
    {
        const bool overlaps = test_node_overlap(path, depth);

        NodeEditDesc out;
        if (!overlaps)
            out.operation = EditOperation::Keep;
        else if (nodeDesc.state == NodeState::Empty)
            out.operation = EditOperation::Empty;
        else
            out.operation = EditOperation::Subdivide;
        return out;
    }

    HOST_DEVICE std::optional<uint32_t> get_new_value(float3 position, std::optional<uint32_t> optOldMaterial) const
    {
        Path path { (unsigned)position.x, (unsigned)position.y, (unsigned)position.z };
        int count = 0;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    // dx/dy/dz might be negative so we need to use signed numbers.
                    Path neighbour = path;
                    neighbour.path.x += dx;
                    neighbour.path.y += dy;
                    neighbour.path.z += dz;
                    count += DAGUtils::get_value(dag, neighbour).has_value();
                }
            }
        }

        if (count > threshold)
            return optOldMaterial;
        else
            return {};
    }
};
