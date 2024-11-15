#include <catch2/catch_all.hpp>
// Include catch2 first.
#include "configuration/gpu_hash_dag_definitions.h"
#include "cuda_math.h" // operator<<
#include "dag_info.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/dag_utils.h"
#include "dags/my_gpu_dags/my_gpu_dag_editors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_edits.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_factory.h"
#include "engine.h"
#include "path.h"
#include "voxel_textures.h"
#include <ostream>
#include <random>

using DAG_CPU = MyGPUHashDAG<EMemoryType::CPU>;
using DAG_GPU = MyGPUHashDAG<EMemoryType::GPU_Malloc>;

struct Box {
    float3 boundsMin, boundsMax;

    bool test_node_overlap(const float3& nodeMin, const float3& nodeMax) const
    {
        return !(boundsMin.x >= nodeMax.x || boundsMin.y >= nodeMax.y || boundsMin.z >= nodeMax.z || boundsMax.x <= nodeMin.x || boundsMax.y <= nodeMin.y || boundsMax.z <= nodeMin.z || nodeMin.x >= boundsMax.x || nodeMin.y >= boundsMax.y || nodeMin.z >= boundsMax.z || nodeMax.x <= boundsMin.x || nodeMax.y <= boundsMin.y || nodeMax.z <= boundsMin.z);
    }

    bool test_node_fully_contained(const float3& nodeMin, const float3& nodeMax) const
    {
        return boundsMin.x <= nodeMin.x && boundsMin.y <= nodeMin.y && boundsMin.z <= nodeMin.z && boundsMax.x >= nodeMax.x && boundsMax.y >= nodeMax.y && boundsMax.z >= nodeMax.z;
    }
};

static uint32_t createBoxDag(DAG_CPU& dag, const Path& path, uint32_t level, const Box& box, uint32_t material)
{
    const uint32_t childLevel = level + 1;
    const uint32_t childDepth = C_maxNumberOfLevels - childLevel;

    if (level == DAG_CPU::leaf_level()) {
        std::array<uint32_t, DAG_CPU::maxLeafSizeInU32> leafBuffer;
        typename DAG_CPU::LeafBuilder leafBuilder { leafBuffer.data() };
        for (uint32_t voxelIdx = 0; voxelIdx < 64; ++voxelIdx) {
            Path voxelPath = path;
            voxelPath.descend((uint8_t)(voxelIdx >> 3));
            voxelPath.descend((uint8_t)(voxelIdx & 0b111));

            const float3 position = voxelPath.as_position(0);
            if (box.test_node_overlap(position, position + 1))
                leafBuilder.set(material);
            leafBuilder.next();
        }
        leafBuilder.finalize();

        if (uint32_t handle = dag.find_leaf(leafBuffer.data()); handle != DAG_CPU::invalid_handle) {
            return handle;
        } else {
            return dag.add_leaf(leafBuffer.data());
        }
    } else {
        std::array<uint32_t, DAG_CPU::maxNodeSizeInU32> nodeBuffer;
        uint32_t childMask = 0;
        for (uint8_t childIdx = 0, childOffset = 0; childIdx < 8; ++childIdx) {
            Path childPath = path;
            childPath.descend(childIdx);

            const float3 boundsMin = childPath.as_position(childDepth);
            const float3 boundsMax = boundsMin + make_float3(float(1u << childDepth));
            if (box.test_node_fully_contained(boundsMin, boundsMax)) {
#if EDITS_ENABLE_MATERIALS
                nodeBuffer[DAG_CPU::get_header_size() + childOffset++] = dag.fullyFilledNodes.read(childLevel, material);
#else
                nodeBuffer[DAG_CPU::get_header_size() + childOffset++] = dag.fullyFilledNodes.read(childLevel, 0);
#endif
                childMask |= 1u << childIdx;
            } else if (box.test_node_overlap(boundsMin, boundsMax)) {
                nodeBuffer[DAG_CPU::get_header_size() + childOffset++] = createBoxDag(dag, childPath, level + 1, box, material);
                childMask |= 1u << childIdx;
            }
        }
        DAG_CPU::encode_node_header(nodeBuffer.data(), childMask);

        if (uint32_t handle = dag.find_node(nodeBuffer.data()); handle != DAG_CPU::invalid_handle) {
            return handle;
        } else {
            return dag.add_node(nodeBuffer.data());
        }
    }
}

TEST_CASE("Create Box In DAG", "[MyGPUHashDAG]")
{
    uint32_t boundsMin = 100, boundsMax = 130;
    const Box box {
        .boundsMin = make_float3((float)boundsMin),
        .boundsMax = make_float3((float)boundsMax)
    };
    const uint32_t material = 0;

    auto myDag = DAG_CPU::allocate(64);
    myDag.firstNodeIndex = createBoxDag(myDag, Path(0, 0, 0), 0, box, material);

    for (uint32_t x = boundsMin - 1; x <= boundsMax + 1; ++x) {
        for (uint32_t y = boundsMin - 1; y <= boundsMax + 1; ++y) {
            for (uint32_t z = boundsMin - 1; z <= boundsMax + 1; ++z) {
                Path path { x, y, z };
                auto optMaterial = DAGUtils::get_value(myDag, path);
                CAPTURE(x, y, z);

                if (x < boundsMin || x >= boundsMax || y < boundsMin || y >= boundsMax || z < boundsMin || z >= boundsMax) {
                    REQUIRE(!optMaterial.has_value());
                } else {
                    REQUIRE(optMaterial.has_value());
                    uint32_t value = optMaterial.value();
                    REQUIRE(value == material);
                }
            }
        }
    }

    myDag.free();
}

TEST_CASE("DAG copy to/from GPU", "[MyGPUHashDAG]")
{
    uint32_t boundsMin = 100, boundsMax = 130;
    const Box box {
        .boundsMin = make_float3((float)boundsMin),
        .boundsMax = make_float3((float)boundsMax)
    };
    const uint32_t material = 0;

    // Construct a DAG on the GPU containing the box.
    DAG_GPU myDagGPU;
    {
        auto myDagCPU = DAG_CPU::allocate(64);
        myDagCPU.firstNodeIndex = createBoxDag(myDagCPU, Path(0, 0, 0), 0, box, material);
        myDagGPU = myDagCPU.copy<EMemoryType::GPU_Malloc>();
        myDagCPU.free();
    }

    // Copy back to the CPU.
    DAG_CPU myDagCPU = myDagGPU.copy<EMemoryType::CPU>();

    // Check if output is correct.
    for (uint32_t x = boundsMin - 1; x <= boundsMax + 1; ++x) {
        for (uint32_t y = boundsMin - 1; y <= boundsMax + 1; ++y) {
            for (uint32_t z = boundsMin - 1; z <= boundsMax + 1; ++z) {
                Path path { x, y, z };
                auto optMaterial = DAGUtils::get_value(myDagCPU, path);
                CAPTURE(x, y, z);

                if (x < boundsMin || x >= boundsMax || y < boundsMin || y >= boundsMax || z < boundsMin || z >= boundsMax) {
                    REQUIRE(!optMaterial.has_value());
                } else {
                    REQUIRE(optMaterial.has_value());
                    uint32_t value = optMaterial.value();
                    REQUIRE(value == material);
                }
            }
        }
    }

    myDagCPU.free();
    myDagGPU.free();
}

TEST_CASE("MyGpuSpherePaintEditor", "[MyGPUHashDAG]")
{
    // Perform editing.
    const uint32_t editMaterial = 60;
    const uint3 center = make_uint3(3599, 8440, 5045);
    const uint32_t radius = 15;
    MyGpuSpherePaintEditor tool { make_float3(center), (float)radius, make_float3(0), editMaterial };

    // Check if output is correct.
    const uint3 boundsMin = center - radius;
    const uint3 boundsMax = center + radius;
    for (uint32_t x = boundsMin.x; x <= boundsMax.x; ++x) {
        for (uint32_t y = boundsMin.y; y <= boundsMax.y; ++y) {
            for (uint32_t z = boundsMin.z; z <= boundsMax.z; ++z) {
                const float3 voxelMin = make_float3((float)x, (float)y, (float)z);
                const float3 voxelMax = voxelMin + make_float3(1.0f);

                const float3 delta = abs(make_float3(center) - voxelMin);
                const float distance = length(delta);
                CAPTURE(x, y, z, center, radius, distance);
                if (distance < radius - 1) {
                    REQUIRE(tool.test_node_overlap(voxelMin, voxelMax));
                } else if (distance > radius + 1) {
                    REQUIRE(!tool.test_node_overlap(voxelMin, voxelMax));
                }
            }
        }
    }
}

TEST_CASE("Paint a sphere inside a box", "[MyGPUHashDAG]")
{
    const uint32_t boundsMin = 100; // GENERATE(100, 10001, 12345);
    const uint32_t boundsMax = boundsMin + 30; // GENERATE(30, 51, 67);
    const Box box {
        .boundsMin = make_float3((float)boundsMin),
        .boundsMax = make_float3((float)boundsMax)
    };
    const uint32_t material = 0;
    const uint32_t editMaterial = 13;
    const uint32_t toolCenterOffsetX = 10; // GENERATE(-10, 0, +10);

    // Construct a DAG on the GPU containing the box.
    DAG_GPU myDagGPU;
    {
        auto myDagCPU = DAG_CPU::allocate(64);
        myDagCPU.firstNodeIndex = createBoxDag(myDagCPU, Path(0, 0, 0), 0, box, material);
        myDagGPU = myDagCPU.copy<EMemoryType::GPU_Malloc>();
        myDagCPU.free();
    }

    // Perform editing.
    const uint3 center = make_uint3((boundsMax + boundsMin) / 2) + make_uint3(toolCenterOffsetX, 0, 0);
    const uint32_t radius = (boundsMax - boundsMin) / 2;
    MyGpuSpherePaintEditor tool { make_float3(center), (float)radius, make_float3(0), editMaterial };

    cudaStream_t stream = nullptr;
    GpuMemoryPool memPool = GpuMemoryPool::create(stream);
    StatsRecorder statsRecorder {};
    MyGPUHashDAGUndoRedo undoRedo;
    editMyHashDag(tool, myDagGPU, undoRedo, statsRecorder, memPool, stream);
    memPool.release();

    // Copy back to the CPU.
    DAG_CPU myDagCPU = myDagGPU.copy<EMemoryType::CPU>();

    // Check if output is correct.
    for (uint32_t x = boundsMin - 1; x <= boundsMax + 1; ++x) {
        for (uint32_t y = boundsMin - 1; y <= boundsMax + 1; ++y) {
            for (uint32_t z = boundsMin - 1; z <= boundsMax + 1; ++z) {
                Path path { x, y, z };
                auto optMaterial = DAGUtils::get_value(myDagCPU, path);
                CAPTURE(x, y, z);

                if (x < boundsMin || x >= boundsMax || y < boundsMin || y >= boundsMax || z < boundsMin || z >= boundsMax) {
                    REQUIRE(!optMaterial.has_value());
                } else { // editMaterial != material
                    REQUIRE(optMaterial.has_value());
                    uint32_t value = optMaterial.value();

                    const auto voxelMin = path.as_position();
                    const auto voxelMax = voxelMin + 1.0f;
                    if (tool.test_node_overlap(voxelMin, voxelMax)) {
                        REQUIRE(value == editMaterial);
                    } else {
                        REQUIRE(value == material);
                    }
                }
            }
        }
    }

    myDagCPU.free();
    myDagGPU.free();
}

/* TEST_CASE("Paint inside epic citadel", "[MyGPUHashDAG]")
{
    const std::filesystem::path rootFolder { ROOT_FOLDER };
    const std::string fileName = std::string(SCENE) + std::to_string(1 << (SCENE_DEPTH - 10)) + "k";

    const std::string gpuHashDagFileName = fileName + ".gpu_hash_dag.dag.bin";
    auto gpuHashDagFilePath = rootFolder / "data" / gpuHashDagFileName;

    auto pEngine = Engine::create();
    BinaryReader reader { gpuHashDagFilePath };
    pEngine->readFrom(reader);

    DAG_CPU myDagCPU = pEngine->myGpuHashDag.copy<EMemoryType::CPU>();

    // Perform editing.
    const uint32_t editMaterial = 3;
    const uint3 center = make_uint3(3599, 8440, 5045);
    const uint32_t radius = 15;
    MyGpuSpherePaintEditor tool { make_float3(center), (float)radius, make_float3(0), editMaterial };

    cudaStream_t stream = nullptr;
    GpuMemoryPool memPool = GpuMemoryPool::create(stream);
    StatsRecorder statsRecorder {};
    for (uint32_t i = 0; i < 1; ++i)
        editMyHashDag(tool, pEngine->myGpuHashDag, pEngine->gpuUndoRedo, statsRecorder, memPool, stream);
    memPool.release();

    // Copy back to the CPU.
    DAG_CPU myEditedDagCPU = pEngine->myGpuHashDag.copy<EMemoryType::CPU>();

    // Check if output is correct.
    const uint3 boundsMin = center - radius;
    const uint3 boundsMax = center + radius;
    int numEdited = 0;
    for (uint32_t x = boundsMin.x; x <= boundsMax.x; ++x) {
        for (uint32_t y = boundsMin.y; y <= boundsMax.y; ++y) {
            for (uint32_t z = boundsMin.z; z <= boundsMax.z; ++z) {
                Path path { x, y, z };
                auto optOriginalMaterial = DAGUtils::get_value(myDagCPU, path);
                auto optEditedMaterial = DAGUtils::get_value(myEditedDagCPU, path);
                CAPTURE(x, y, z);

                REQUIRE(optOriginalMaterial.has_value() == optEditedMaterial.has_value());
                if (optOriginalMaterial.has_value()) {
                    const uint32_t originalMaterial = optOriginalMaterial.value();
                    const uint32_t editedMaterial = optEditedMaterial.value();
                    CAPTURE(originalMaterial, editedMaterial);


                    const auto voxelMin = path.as_position();
                    const auto voxelMax = voxelMin + 1.0f;
                    if (tool.test_node_overlap(voxelMin, voxelMax)) {
                        ++numEdited;
                        REQUIRE(editedMaterial == editMaterial);
                    } else {
                        REQUIRE(editedMaterial == originalMaterial);
                    }
                }
            }
        }
    }

    myDagCPU.free();
    myEditedDagCPU.free();
    pEngine->destroy();
} */