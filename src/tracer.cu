#include "dags/basic_dag/basic_dag.h"
#include "dags/dag_utils.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_colors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "tracer.h"
#include "utils.h"
#include <cmath>
#include <cuda_math.h>
#include <numbers>

namespace Tracer {

template <typename T>
struct dag_has_materials : std::false_type {
};
#if EDITS_ENABLE_MATERIALS
template <>
struct dag_has_materials<MyGPUHashDAG<EMemoryType::GPU_Malloc>> : std::true_type {
};
// template <>
// struct dag_has_materials<MyGpuSortedDag> : std::true_type {
// };
#endif
template <typename T>
static constexpr bool dag_has_materials_v = dag_has_materials<T>::value;

struct Ray {
    float3 origin, direction, invDirection;
    float tmin, tmax;

    static HOST_DEVICE Ray create(float3 origin, float3 direction)
    {
        return {
            .origin = origin,
            .direction = direction,
            .invDirection = 1.0f / direction,
            .tmin = 0.0f,
            .tmax = std::numeric_limits<float>::max()
        };
    }
};

// order: (shouldFlipX, shouldFlipY, shouldFlipZ)
DEVICE uint8 next_child(uint8 order, uint8 mask)
{
    for (uint8 child = 0; child < 8; ++child) {
        uint8 childInOrder = child ^ order;
        if (mask & (1u << childInOrder))
            return childInOrder;
    }
    check(false);
    return 0;
}

template <bool isRoot, typename TDAG>
DEVICE uint8 compute_intersection_mask(uint32 level, const Path& path, const TDAG& dag, const Ray ray, float& tmin)
{
    // Find node center = .5 * (boundsMin + boundsMax) + .5f
    const uint32 shift = dag.levels - level;

    const float radius = float(1u << (shift - 1));
    const float3 center = make_float3(radius) + path.as_position(shift);

    const float3 centerRelativeToRay = center - ray.origin;

    // Ray intersection with axis-aligned planes centered on the node
    // => rayOrg + tmid * rayDir = center
    const float3 tmid = centerRelativeToRay * ray.invDirection;

    // t-values for where the ray intersects the slabs centered on the node
    // and extending to the side of the node
    float tmax;
    {
        const float3 slabRadius = radius * abs(ray.invDirection);
        const float3 pmin = tmid - slabRadius;
        tmin = max(max(pmin), 0.0f);

        const float3 pmax = tmid + slabRadius;
        tmax = min(pmax);
    }
    tmin = max(tmin, ray.tmin);
    tmax = min(tmax, ray.tmax);

    // Check if we actually hit the root node
    // This test may not be entirely safe due to float precision issues.
    // especially on lower levels. For the root node this seems OK, though.
    if (isRoot && (tmin >= tmax)) {
        return 0;
    }

    // Identify first child that is intersected
    // NOTE: We assume that we WILL hit one child, since we assume that the
    //       parents bounding box is hit.
    // NOTE: To safely get the correct node, we cannot use o+ray_tmin*d as the
    //       intersection point, since this point might lie too close to an
    //       axis plane. Instead, we use the midpoint between max and min which
    //       will lie in the correct node IF the ray only intersects one node.
    //       Otherwise, it will still lie in an intersected node, so there are
    //       no false positives from this.
    uint8 intersectionMask = 0;
    {
        const float3 pointOnRay = (0.5f * (tmin + tmax)) * ray.direction;

        uint8 const firstChild = ((pointOnRay.x >= centerRelativeToRay.x) ? 4 : 0) + ((pointOnRay.y >= centerRelativeToRay.y) ? 2 : 0) + ((pointOnRay.z >= centerRelativeToRay.z) ? 1 : 0);

        intersectionMask |= (1u << firstChild);
    }

    // We now check the points where the ray intersects the X, Y and Z plane.
    // If the intersection is within (ray_tmin, ray_tmax) then the intersection
    // point implies that two voxels will be touched by the ray. We find out
    // which voxels to mask for an intersection point at +X, +Y by setting
    // ALL voxels at +X and ALL voxels at +Y and ANDing these two masks.
    //
    // NOTE: When the intersection point is close enough to another axis plane,
    //       we must check both sides or we will get robustness issues.
    const float epsilon = 1e-4f;

    if (tmin <= tmid.x && tmid.x <= tmax) {
        const float3 pointOnRay = tmid.x * ray.direction;

        uint8 A = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            A |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            A |= 0x33;

        uint8 B = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            B |= 0xAA;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            B |= 0x55;

        intersectionMask |= A & B;
    }
    if (tmin <= tmid.y && tmid.y <= tmax) {
        const float3 pointOnRay = tmid.y * ray.direction;

        uint8 C = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            C |= 0xF0;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            C |= 0x0F;

        uint8 D = 0;
        if (pointOnRay.z >= centerRelativeToRay.z - epsilon)
            D |= 0xAA;
        if (pointOnRay.z <= centerRelativeToRay.z + epsilon)
            D |= 0x55;

        intersectionMask |= C & D;
    }
    if (tmin <= tmid.z && tmid.z <= tmax) {
        const float3 pointOnRay = tmid.z * ray.direction;

        uint8 E = 0;
        if (pointOnRay.x >= centerRelativeToRay.x - epsilon)
            E |= 0xF0;
        if (pointOnRay.x <= centerRelativeToRay.x + epsilon)
            E |= 0x0F;

        uint8 F = 0;
        if (pointOnRay.y >= centerRelativeToRay.y - epsilon)
            F |= 0xCC;
        if (pointOnRay.y <= centerRelativeToRay.y + epsilon)
            F |= 0x33;

        intersectionMask |= E & F;
    }

    return intersectionMask;
}
template <bool isRoot, typename TDAG>
DEVICE uint8 compute_intersection_mask(uint32 level, const Path& path, const TDAG& dag, const Ray ray)
{
    float tmin;
    return compute_intersection_mask<isRoot, TDAG>(level, path, dag, ray, tmin);
}

struct StackEntry {
    uint32 index;
    uint8 childMask;
    uint8 visitMask;
};
template <typename TDAG>
DEVICE Path intersect_ray(const TDAG& dag, Ray& ray, uint32_t& materialId)
{
    const uint8 rayChildOrder = (ray.direction.x < 0.f ? 4 : 0) + (ray.direction.y < 0.f ? 2 : 0) + (ray.direction.z < 0.f ? 1 : 0);

    // State
    uint32_t level = 0, addr;
    Path path(0, 0, 0);

    StackEntry stack[MAX_LEVELS];
    StackEntry cache;
    Leaf cachedLeaf; // needed to iterate on the last few levels

    cache.index = dag.get_first_node_index();
    cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
    cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, ray);
    float tmin;

    // Traverse DAG
    for (;;) {
        // Ascend if there are no children left.
        {
            uint32 newLevel = level;
            while (newLevel > 0 && !cache.visitMask) {
                newLevel--;
                cache = stack[newLevel];
            }

            if (newLevel == 0 && !cache.visitMask) {
                path = Path(0, 0, 0);
                break;
            }

            path.ascend(level - newLevel);
            level = newLevel;
        }

        // Find next child in order by the current ray's direction
        const uint8 nextChild = next_child(rayChildOrder, cache.visitMask);

        // Mark it as handled
        cache.visitMask &= ~(1u << nextChild);

        // Intersect that child with the ray
        {
            path.descend(nextChild);
            stack[level] = cache;
            level++;

            // If we're at the final level, we have intersected a single voxel.
            if (level == dag.levels) {
                if constexpr (dag_has_materials_v<TDAG>) {
                    const auto voxelIdx = path.mortonU32() & 0b111111;
                    if (!dag.get_material(dag.get_leaf_ptr(addr), voxelIdx, materialId))
                        materialId = 0xFFFFFFFF;
                }
                ray.tmin = tmin;
                break;
            }

            // Are we in an internal node?
            if (level < dag.leaf_level()) {
                cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, ray);
            } else {
                /* The second-to-last and last levels are different: the data
                 * of these two levels (2^3 voxels) are packed densely into a
                 * single 64-bit word.
                 */
                uint8 childMask;

                if (level == dag.leaf_level()) {
                    addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                    cachedLeaf = dag.get_leaf(addr);
                    childMask = cachedLeaf.get_first_child_mask();
                } else {
                    childMask = cachedLeaf.get_second_child_mask(nextChild);
                }

                // No need to set the index for bottom nodes
                cache.childMask = childMask;
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, ray);
            }
        }
    }
    return path;
}
template <typename TDAG>
DEVICE bool intersect_ray_node_out_of_order(const TDAG& dag, Ray ray)
{
    // State
    uint32 level = 0;
    Path path(0, 0, 0);

    StackEntry stack[MAX_LEVELS];
    StackEntry cache;
    Leaf cachedLeaf; // needed to iterate on the last few levels

    cache.index = dag.get_first_node_index();
    cache.childMask = Utils::child_mask(dag.get_node(0, cache.index));
    cache.visitMask = cache.childMask & compute_intersection_mask<true>(0, path, dag, ray);

    // Traverse DAG
    for (;;) {
        // Ascend if there are no children left.
        {
            uint32 newLevel = level;
            while (newLevel > 0 && !cache.visitMask) {
                newLevel--;
                cache = stack[newLevel];
            }

            if (newLevel == 0 && !cache.visitMask) {
                path = Path(0, 0, 0);
                break;
            }

            path.ascend(level - newLevel);
            level = newLevel;
        }

        // Find next child in order by the current ray's direction
        const uint8 nextChild = 31 - __clz(cache.visitMask);

        // Mark it as handled
        cache.visitMask &= ~(1u << nextChild);

        // Intersect that child with the ray
        {
            path.descend(nextChild);
            stack[level] = cache;
            level++;

            // If we're at the final level, we have intersected a single voxel.
            if (level == dag.levels) {
                return true;
            }

            // Are we in an internal node?
            if (level < dag.leaf_level()) {
                cache.index = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                cache.childMask = Utils::child_mask(dag.get_node(level, cache.index));
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, ray);
            } else {
                /* The second-to-last and last levels are different: the data
                 * of these two levels (2^3 voxels) are packed densely into a
                 * single 64-bit word.
                 */
                uint8 childMask;

                if (level == dag.leaf_level()) {
                    const uint32 addr = dag.get_child_index(level - 1, cache.index, cache.childMask, nextChild);
                    cachedLeaf = dag.get_leaf(addr);
                    childMask = cachedLeaf.get_first_child_mask();
                } else {
                    childMask = cachedLeaf.get_second_child_mask(nextChild);
                }

                // No need to set the index for bottom nodes
                cache.childMask = childMask;
                cache.visitMask = cache.childMask & compute_intersection_mask<false>(level, path, dag, ray);
            }
        }
    }
    return false;
}
HOST_DEVICE SurfaceInteraction createSurfaceInteraction(const Ray ray, const Path& path, uint8_t materialId)
{
    // Find the face and UV coordinates of the voxel/ray intersection.
    const float3 boundsMin = path.as_position();
    const float3 boundsMax = boundsMin + make_float3(1.0f);
    const float3 t1 = (boundsMin - ray.origin) * ray.invDirection;
    const float3 t2 = (boundsMax - ray.origin) * ray.invDirection;

    const float3 dists = min(t1, t2);
    const int axis = dists.x > dists.y ? (dists.x > dists.z ? 0 : 2) : (dists.y > dists.z ? 1 : 2);

    float t;
    SurfaceInteraction out;
    out.path = path.path;
    out.materialId = materialId;
    out.normal = make_float3(0.0f);
    out.dpdu = make_float3(0.0f);
    out.dpdv = make_float3(0.0f);
    if (axis == 0) {
        out.normal.x = (ray.direction.x < 0.0f ? +1.0f : -1.0f);
        out.dpdu.z = 1.0f;
        out.dpdv.y = 1.0f;
        t = dists.x;
    } else if (axis == 1) {
        out.normal.y = (ray.direction.y < 0.0f ? +1.0f : -1.0f);
        out.dpdu.x = 1.0f;
        out.dpdv.z = 1.0f;
        t = dists.y;
    } else if (axis == 2) {
        out.normal.z = (ray.direction.z < 0.0f ? +1.0f : -1.0f);
        out.dpdu.x = 1.0f;
        out.dpdv.y = 1.0f;
        t = dists.z;
    }
    out.position = ray.origin + t * ray.direction;
    return out;
}

template <typename TDAG>
__global__ void trace_paths(TracePathsParams traceParams, const TDAG dag)
{
    // Target pixel coordinate
    const uint2 pixel = make_uint2(
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside.

    // Pre-calculate per-pixel data
    const float3 rayOrigin = make_float3(traceParams.cameraPosition);
    const float3 rayDirection = make_float3(normalize(traceParams.rayMin + pixel.x * traceParams.rayDDx + pixel.y * traceParams.rayDDy - traceParams.cameraPosition));
    Ray ray = Ray::create(rayOrigin, rayDirection);

    uint32_t materialId;
    const auto path = intersect_ray(dag, ray, materialId);

    traceParams.surfaceInteractionSurface.write(pixel.x, imageHeight - 1 - pixel.y, createSurfaceInteraction(ray, path, materialId));
}

template <typename TDAG, typename TDAGColors>
DEVICE float3 computeColorAtSurface(const ColorParams& colorParams, const TDAG dag, const TDAGColors colors, const SurfaceInteraction& si)
{
    Path path = si.path;
    if (path.is_null())
        return make_float3(0.0f);

    const auto invalidColor = [&]() {
        uint32 b = (path.path.x ^ path.path.y ^ path.path.z) & 0x1;
        return make_float3(1, b, 1.f - b);
    };

    if constexpr (dag_has_materials_v<TDAG>) { // Materials
#if EDITS_ENABLE_MATERIALS
        const auto materialId = si.materialId;
        if (materialId >= colorParams.materialTextures.size())
            return invalidColor();

        const int axis = std::abs(si.normal.x) > 0.5f ? 0 : (std::abs(si.normal.y) > 0.5f ? 1 : 2);
        const float3 offset = si.position - Path(path).as_position();
        float2 uv;
        if (axis == 0) {
            uv = make_float2(offset.z, offset.y);
        } else if (axis == 1) {
            uv = make_float2(offset.x, offset.z);
        } else { // axis == 2
            uv = make_float2(offset.x, offset.y);
        }

        const auto& material = colorParams.materialTextures[materialId];
        auto topPath = path;
        topPath.path.y += 1;
        cudaTextureObject_t texture;
        if (axis == 1 && !material.all.cuArray) {
            texture = material.top.cuTexture;
        } else if (axis == 1 && si.normal.y > 0 && material.top.cuArray && DAGUtils::get_value(dag, topPath)) {
            texture = material.top.cuTexture;
        } else if (axis != 1 && material.side.cuArray) {
            texture = material.side.cuTexture;
        } else {
            texture = material.all.cuTexture;
        }

        const auto tmp = tex2D<float4>(texture, uv.x, 1 - uv.y);
        return make_float3(tmp.x, tmp.y, tmp.z);
#else
        invalidColor();
#endif

    } else { // Materials vs colors
        if (!colors.is_valid())
            return invalidColor();

        uint64 nof_leaves = 0;
        uint32 debugColorsIndex = 0;

        uint32 colorNodeIndex = 0;
        typename TDAGColors::ColorLeaf colorLeaf = colors.get_default_leaf();

        uint32 level = 0;
        uint32 nodeIndex = dag.get_first_node_index();
        while (level < dag.leaf_level()) {
            level++;

            // Find the current childmask and which subnode we are in
            const uint32 node = dag.get_node(level - 1, nodeIndex);
            const uint8 childMask = Utils::child_mask(node);
            const uint8 child = path.child_index(level, dag.levels);

            // Make sure the node actually exists
            if (!(childMask & (1 << child)))
                return make_float3(1.0f, 0.0f, 1.0f);

            ASSUME(level > 0);
            if (level - 1 < colors.get_color_tree_levels()) {
                colorNodeIndex = colors.get_child_index(level - 1, colorNodeIndex, child);
                if (level == colors.get_color_tree_levels()) {
                    check(nof_leaves == 0);
                    colorLeaf = colors.get_leaf(colorNodeIndex);
                } else {
                    // TODO nicer interface
                    if (!colorNodeIndex)
                        return invalidColor();
                }
            }

            // Debug
            if (colorParams.debugColors == EDebugColors::Index || colorParams.debugColors == EDebugColors::Position || colorParams.debugColors == EDebugColors::ColorTree) {
                if (colorParams.debugColors == EDebugColors::Index && colorParams.debugColorsIndexLevel == level - 1) {
                    debugColorsIndex = nodeIndex;
                }
                if (level == dag.leaf_level()) {
                    if (colorParams.debugColorsIndexLevel == dag.leaf_level()) {
                        check(debugColorsIndex == 0);
                        const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
                        debugColorsIndex = childIndex;
                    }

                    if (colorParams.debugColors == EDebugColors::Index) {
                        return ColorUtils::rgb888_to_float3(Utils::murmurhash32(debugColorsIndex));
                    } else if (colorParams.debugColors == EDebugColors::Position) {
                        constexpr uint32 checkerSize = 0x7FF;
                        float color = ((path.path.x ^ path.path.y ^ path.path.z) & checkerSize) / float(checkerSize);
                        color = (color + 0.5) / 2;
                        return Utils::has_flag(nodeIndex) ? make_float3(color, 0, 0) : make_float3(color);
                    } else {
                        check(colorParams.debugColors == EDebugColors::ColorTree);
                        const uint32 offset = dag.levels - colors.get_color_tree_levels();
                        const float color = ((path.path.x >> offset) ^ (path.path.y >> offset) ^ (path.path.z >> offset)) & 0x1;
                        return make_float3(color);
                    }
                } else {
                    nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
                    continue;
                }
            }

            //////////////////////////////////////////////////////////////////////////
            // Find out how many leafs are in the children preceding this
            //////////////////////////////////////////////////////////////////////////
            // If at final level, just count nof children preceding and exit
            if (level == dag.leaf_level()) {
                for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild) {
                    if (childMask & (1u << childBeforeChild)) {
                        const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
                        const Leaf leaf = dag.get_leaf(childIndex);
                        nof_leaves += Utils::popcll(leaf.to_64());
                    }
                }
                const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
                const Leaf leaf = dag.get_leaf(childIndex);
                const uint8 leafBitIndex = (((path.path.x & 0x1) == 0) ? 0 : 4) | (((path.path.y & 0x1) == 0) ? 0 : 2) | (((path.path.z & 0x1) == 0) ? 0 : 1) | (((path.path.x & 0x2) == 0) ? 0 : 32) | (((path.path.y & 0x2) == 0) ? 0 : 16) | (((path.path.z & 0x2) == 0) ? 0 : 8);
                nof_leaves += Utils::popcll(leaf.to_64() & ((uint64(1) << leafBitIndex) - 1));

                break;
            } else {
                ASSUME(level > 0);
                if (level > colors.get_color_tree_levels()) {
                    // Otherwise, fetch the next node (and accumulate leaves we pass by)
                    for (uint8 childBeforeChild = 0; childBeforeChild < child; ++childBeforeChild) {
                        if (childMask & (1u << childBeforeChild)) {
                            const uint32 childIndex = dag.get_child_index(level - 1, nodeIndex, childMask, childBeforeChild);
                            // if constexpr (std::is_same_v<TDAG, MyHashDag> || std::is_same_v<TDAG, MyGPUHashDAG<EMemoryType::GPU_Malloc>> || std::is_same_v<TDAG, MyGpuSortedDag>)
                            if constexpr (std::is_same_v<TDAG, MyGPUHashDAG<EMemoryType::GPU_Malloc>>)
                                nof_leaves += 0;
                            else
                                nof_leaves += colors.get_leaves_count(level, dag.get_node(level, childIndex));
                        }
                    }
                }
                nodeIndex = dag.get_child_index(level - 1, nodeIndex, childMask, child);
            }
        }

        if (!colorLeaf.is_valid() || !colorLeaf.is_valid_index(nof_leaves)) {
            return invalidColor();
        }

        const auto compressedColor = colorLeaf.get_color(nof_leaves);
        if (colorParams.debugColors == EDebugColors::ColorBits) {
            return ColorUtils::rgb888_to_float3(compressedColor.get_debug_hash());
        } else if (colorParams.debugColors == EDebugColors::MinColor) {
            return compressedColor.get_min_color();
        } else if (colorParams.debugColors == EDebugColors::MaxColor) {
            return compressedColor.get_max_color();
        } else if (colorParams.debugColors == EDebugColors::Weight) {
            return make_float3(compressedColor.get_weight());
        } else {
            return compressedColor.get_color();
        }
    }
}

template <typename TDAG, typename TDAGColors>
__global__ void trace_colors(TraceColorsParams traceParams, const TDAG dag, const TDAGColors colors)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    SurfaceInteraction& si = traceParams.surfaceInteractionSurface.getPixel(pixel);
    si.diffuseColorSRGB8 = ColorUtils::float3_to_rgb888(
        computeColorAtSurface(traceParams.colorParams, dag, colors, si));
}

// Directed towards the sun
HOST_DEVICE float3 sun_direction()
{
    return normalize(make_float3(0.3f, 1.f, 0.5f));
}
HOST_DEVICE float3 sun_color()
{
    return make_float3(0.95f, 0.94f, 0.47f); // White/yellow;
}
HOST_DEVICE float3 sky_color()
{
    return make_float3(187, 242, 250) / 255.f; // blue
}

template <typename TDAG>
__global__ void trace_shadows(TraceShadowsParams params, const TDAG dag)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const SurfaceInteraction& si = params.surfaceInteractionSurface.getPixel(pixel);
    if (si.path.is_null())
        return;

    const Ray shadowRay = Ray::create(si.position + params.shadowBias * sun_direction(), sun_direction());
    const bool sunOccluded = intersect_ray_node_out_of_order(dag, shadowRay);
    const float NdotL = std::max(dot(si.normal, shadowRay.direction), 0.0f);
    params.sunLightSurface.write(pixel, sunOccluded ? 0.0f : NdotL);
}

// https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
struct RandomSample {
    float3 direction;
    float pdf;
};
HOST_DEVICE RandomSample uniformSampleHemisphere(const float2& u)
{
    const auto z = u.x;
    const auto r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const auto phi = 2 * std::numbers::pi_v<float> * u.y;
    return {
        .direction = make_float3(r * std::cos(phi), r * std::sin(phi), z),
        .pdf = 0.5f * std::numbers::inv_pi_v<float>
    };
}
HOST_DEVICE RandomSample cosineSampleHemisphere(const float2& u)
{
    // https://www.rorydriscoll.com/2009/01/07/better-sampling/
    const float r = std::sqrt(u.x);
    const float theta = 2 * std::numbers::pi_v<float> * u.y;

    const float x = r * std::cos(theta);
    const float y = r * std::sin(theta);
    const float z = std::sqrt(std::max(0.0f, 1 - u.x));

    return {
        .direction = make_float3(x, y, z),
        .pdf = z * std::numbers::inv_pi_v<float>
    };
}

template <typename TDAG>
__global__ void trace_ambient_occlusion(TraceAmbientOcclusionParams traceParams, const TDAG dag)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const SurfaceInteraction& si = traceParams.surfaceInteractionSurface.getPixel(pixel);
    if (si.path.is_null())
        return;

    const uint64_t seed = (pixel.y * imageWidth + pixel.x) * std::max(traceParams.randomSeed, (uint64_t)1);
    Utils::PCG_RNG rng { seed };
    uint32_t numSamples = 0, numHits = 0;
    for (uint32_t i = 0; i < traceParams.numSamples; ++i) {
        const float2 u = rng.sampleFloat2();
        const auto hemisphereSample = cosineSampleHemisphere(u);
        const float3 worldDirection = si.transformDirectionToWorld(hemisphereSample.direction);
        Ray shadowRay = Ray::create(si.position + si.normal * traceParams.shadowBias, worldDirection);
        shadowRay.tmax = traceParams.aoRayLength;

        const float NdotL = max(dot(si.normal, shadowRay.direction), 0.0f);
        if (!intersect_ray_node_out_of_order(dag, shadowRay)) {
            numHits += 1;
        }
        numSamples += 1;
    }

    const float ambientOcclusion = (float)numHits / (float)numSamples;
    traceParams.aoSurface.write(pixel.x, pixel.y, ambientOcclusion);
}

HOST_DEVICE float3 applyFog(
    float3 rgb, // original color of the pixel
    const Ray ray,
    const SurfaceInteraction si,
    float fogDensity)
{
    fogDensity *= 0.00001f;
    const float fogAmount = 1.0f - exp(-length(ray.origin - si.position) * fogDensity);
    const float sunAmount = 1.01f * max(dot(ray.direction, sun_direction()), 0.0f);
    const float3 fogColor = lerp(
        sky_color(), // blue
        make_float3(1.0f), // white
        pow(sunAmount, 30.0f));
    return lerp(rgb, fogColor, clamp(fogAmount, 0.0f, 1.0f));
}

__global__ void trace_lighting(TraceLightingParams params)
{
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside

    const SurfaceInteraction& si = params.surfaceInteractionSurface.getPixel(pixel);
    float3 shadedColor;
    if (si.path.is_null()) {
        shadedColor = sky_color();
    } else {
        const float3 materialKd_sRGB = ColorUtils::rgb888_to_float3(si.diffuseColorSRGB8);
        const float3 materialKd = ColorUtils::accurateSRGBToLinear(materialKd_sRGB);

        constexpr float ambientColorFactor = 0.5f;

        const float ambientOcclusion = params.applyAmbientOcclusion ? params.aoSurface.read(pixel) : 1.0f;
        const float sunVisibility = params.applyShadows ? params.sunLightSurface.read(pixel) : 1.0f;
        const float3 ambientLightContribution = make_float3(ambientOcclusion) * ambientColorFactor * sky_color();
        const float3 directLightContribution = sunVisibility * sun_color() * params.sunBrightness;
        shadedColor = (ambientLightContribution + directLightContribution) * materialKd;

        if (params.fogDensity > 0.0f) {
            // Reconstruct camera ray.
            const float3 rayOrigin = make_float3(params.cameraPosition);
            const float3 rayDirection = make_float3(normalize(params.rayMin + pixel.x * params.rayDDx + pixel.y * params.rayDDy - params.cameraPosition));
            const Ray cameraRay = Ray::create(rayOrigin, rayDirection);
            shadedColor = applyFog(shadedColor, cameraRay, si, params.fogDensity);
        }
    }

#ifdef TOOL_OVERLAY
    if (!si.path.is_null())
        shadedColor = params.toolInfo.addToolColor(si.path, shadedColor);
    shadedColor = min(shadedColor, make_float3(1.0f));
#endif

    const uint32 finalColor = ColorUtils::float3_to_rgb888(
        ColorUtils::accurateLinearToSRGB(shadedColor));
    params.finalColorsSurface.write(pixel.x, pixel.y, finalColor);
}

[[maybe_unused]] DEVICE static float getColorLuminance(float3 linear)
{
    const float3 sRGB = ColorUtils::approximationLinearToSRGB(linear);
    return 0.299f * sRGB.x + 0.587f * sRGB.y * 0.114 * sRGB.z;
}

template <typename TDAG, typename TDAGColors>
__global__ void trace_path_tracing(TracePathTracingParams traceParams, const TDAG dag, const TDAGColors colors)
{
    // Target pixel coordinate
    const uint2 pixel = make_uint2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (pixel.x >= imageWidth || pixel.y >= imageHeight)
        return; // outside.

    // Epsilon to prevent self intersection.
    constexpr float epsilon = 0.1f;

    // Create camera ray
    const uint64_t seed = (pixel.y * imageWidth + pixel.x) * std::max(traceParams.randomSeed, (uint64_t)1);
    Utils::PCG_RNG rng { seed };
    float3 Lo = traceParams.accumulationBuffer.read(pixel);
    for (uint32_t sample = 0; sample < traceParams.numSamples; ++sample) {
        const float2 cameraSample = make_float2(pixel.x, pixel.y) + (traceParams.integratePixel ? rng.sampleFloat2() : make_float2(0.0f, 0.0f));
        const float3 cameraRayOrigin = make_float3(traceParams.cameraPosition);
        const float3 cameraRayDirection = make_float3(normalize(traceParams.rayMin + cameraSample.x * traceParams.rayDDx + (imageHeight - 1 - cameraSample.y) * traceParams.rayDDy - traceParams.cameraPosition));
        Ray continuationRay = Ray::create(cameraRayOrigin, cameraRayDirection);

        float3 currentPathContribution = make_float3(1.0f);
        for (uint32_t pathDepth = 0; pathDepth < traceParams.maxPathDepth; ++pathDepth) {
            SurfaceInteraction si;
            float3 materialKd;
            if (pathDepth == 0 && !traceParams.integratePixel) {
                si = traceParams.surfaceInteractionSurface.read(pixel);
                materialKd = ColorUtils::accurateSRGBToLinear(ColorUtils::rgb888_to_float3(si.diffuseColorSRGB8));
            } else {
                uint32_t materialId;
                const Path path = intersect_ray(dag, continuationRay, materialId);
                if (!path.is_null()) {
                    si = createSurfaceInteraction(continuationRay, path, materialId);
                    materialKd = ColorUtils::accurateSRGBToLinear(computeColorAtSurface(traceParams.colorParams, dag, colors, si));
                } else {
                    si.path = path; // Store the path so the SurfaceInteraction is considered invalid (and skylighting is used instead).
                }
            }

            // Stop recursion if we didn't hit any geometry (ray goes into the skybox).
            if (si.path.is_null()) {
                Lo = Lo + currentPathContribution * sky_color() * traceParams.environmentBrightness;
                break;
            }

            const auto evaluteBRDF = [&](const float3& direction) {
                const float NdotL = max(0.0f, dot(si.normal, direction));
                return NdotL * std::numbers::inv_pi_v<float> * materialKd;
            };

            // Shadow ray (Next-Event Estimation).
            const Ray shadowRay = Ray::create(si.position + epsilon * si.normal, sun_direction());
            if (!intersect_ray_node_out_of_order(dag, shadowRay)) {
                const float3 brdf = evaluteBRDF(shadowRay.direction);
                Lo = Lo + currentPathContribution * brdf * sun_color() * traceParams.environmentBrightness;
            }

            // Continuation ray.
            auto continuationSample = uniformSampleHemisphere(rng.sampleFloat2());
            const float3 continuationDirection = si.transformDirectionToWorld(continuationSample.direction);
            continuationRay = Ray::create(si.position + epsilon * si.normal, continuationDirection);
            const float3 brdf = evaluteBRDF(continuationRay.direction);
            currentPathContribution = currentPathContribution * (brdf / continuationSample.pdf);

            /*// Russian Roulette.
            const float russianRouletteProbability = clamp(getColorLuminance(currentPathContribution), 0.01f, 1.0f); // Must be in the range of 0 to 1
            if (rng.sampleFloat() > russianRouletteProbability)
                break;
            currentPathContribution = currentPathContribution / russianRouletteProbability; */
        }
    }

    traceParams.accumulationBuffer.write(pixel, Lo);
    Lo = Lo / make_float3(float(traceParams.numSamples + traceParams.numAccumulatedSamples));

#ifdef TOOL_OVERLAY
    const Path cameraPath = traceParams.surfaceInteractionSurface.read(pixel).path;
    if (!cameraPath.is_null())
        Lo = traceParams.toolInfo.addToolColor(cameraPath, Lo);
#endif

    Lo = min(Lo, make_float3(1.0f));
    const uint32_t finalColor = ColorUtils::float3_to_rgb888(ColorUtils::accurateLinearToSRGB(Lo));
    traceParams.finalColorsSurface.write(pixel, finalColor);
}

#define DAG_IMPLS(Dag)                                                    \
    template __global__ void trace_paths<Dag>(TracePathsParams, Dag);     \
    template __global__ void trace_shadows<Dag>(TraceShadowsParams, Dag); \
    template __global__ void trace_ambient_occlusion<Dag>(TraceAmbientOcclusionParams, Dag);

#define DAG_COLOR_IMPLS(Dag, Colors)                                                    \
    template __global__ void trace_colors<Dag, Colors>(TraceColorsParams, Dag, Colors); \
    template __global__ void trace_path_tracing<Dag, Colors>(TracePathTracingParams, Dag, Colors);

// Work-around for not supporting comma in #define (which don't understand C++ syntax) :-(
using MyGPUHashDAG_T = MyGPUHashDAG<EMemoryType::GPU_Malloc>;

DAG_IMPLS(BasicDAG)
DAG_IMPLS(HashDAG)
DAG_IMPLS(MyGPUHashDAG_T)

DAG_COLOR_IMPLS(BasicDAG, BasicDAGUncompressedColors)
DAG_COLOR_IMPLS(BasicDAG, BasicDAGCompressedColors)
DAG_COLOR_IMPLS(BasicDAG, BasicDAGColorErrors)
DAG_COLOR_IMPLS(HashDAG, HashDAGColors)
DAG_COLOR_IMPLS(MyGPUHashDAG_T, HashDAGColors)

} // namespace Tracer
