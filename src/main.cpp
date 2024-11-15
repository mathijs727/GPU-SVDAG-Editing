// #define ENABLE_CHECKS 1
#include "color_utils.h"
#include "configuration/gpu_hash_dag_definitions.h"
#include "configuration/profile_definitions.h"
#include "configuration/script_definitions.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/dag_utils.h"
#include "dags/hash_dag/hash_dag_editors.h"
#include "dags/hash_dag/hash_dag_factory.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_factory.h"
#include "engine.h"
#include "memory.h"
#include "test_shared_memory.h"
#include "typedefs.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <magic_enum.hpp>
#include <nfd.h>
// #include <tbb/task_arena.h>
#ifdef _MSC_VER
#pragma warning(disable : 4702) // Shut up about unreachable code in main
#endif

static const std::filesystem::path rootFolder { ROOT_FOLDER };

// #undef DAG_TYPE
// #define DAG_TYPE EDag::HashDag

// #undef REPLAY_NAME
// #define REPLAY_NAME "large_edits"

/* #undef USE_REPLAY
#define USE_REPLAY 0

#undef REPLAY_NAME
#define REPLAY_NAME "large_edits"

#undef DAG_TYPE
#define DAG_TYPE EDag::MyGpuHashDagLossless888

#undef SCENE
#define SCENE "epiccitadel" */

int main(int argc, char** argv)
{
    PROFILE_FUNCTION();

    auto pEngine = Engine::create();
    pEngine->init(HEADLESS);
    CUDA_CHECK_ERROR();

    printf("Using " SCENE "\n");
    printf("%d levels (resolution=%d^3)\n", MAX_LEVELS, 1 << MAX_LEVELS);
#if ENABLE_CHECKS
    std::fprintf(stderr, "CHECKS: ENABLED\n");
#else
    printf("CHECKS: DISABLED\n");
#endif
    printf("IMAGE RESOLUTION: %ux%u\n", imageWidth, imageHeight);

    const std::string fileName = std::string(SCENE) + std::to_string(1 << (SCENE_DEPTH - 10)) + "k";

    std::string gpuHashDagFileName = fileName;
    if (EDITS_ENABLE_MATERIALS)
        gpuHashDagFileName += ".mat" + std::to_string(HASH_DAG_MATERIAL_BITS);
    else
        gpuHashDagFileName += ".nomat";
#if __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
    gpuHashDagFileName += "." + std::string(magic_enum::enum_name(HASH_TABLE_TYPE));
#pragma GCC diagnostic pop
#else
    gpuHashDagFileName += "." + std::string(magic_enum::enum_name(HASH_TABLE_TYPE));
#endif
    gpuHashDagFileName += ".gpu_hash_dag.dag.bin";

    auto gpuHashDagFilePath = rootFolder / "data" / gpuHashDagFileName;
    const auto basicDagFilePath = rootFolder / "data" / (fileName + ".basic_dag.dag.bin");
    const auto uncompressedColorsFilePath = rootFolder / "data" / (fileName + ".basic_dag.uncompressed_colors.bin");
    const auto compressedColorsFilePath = rootFolder / "data" / (fileName + ".basic_dag.compressed_colors.variable.bin");

#if defined(SAVE_SCENE) && SAVE_SCENE
    if (std::filesystem::exists(gpuHashDagFilePath))
        std::filesystem::remove_all(gpuHashDagFilePath);
#endif // SAVE_SCENE

    if (!std::filesystem::exists(gpuHashDagFilePath) && !std::filesystem::exists(basicDagFilePath)) {
        nfdchar_t* outPathCharArray = nullptr;
        nfdresult_t result = NFD_OpenDialog("gpu_hash_dag.dag.bin", nullptr, &outPathCharArray);

        if (result == NFD_OKAY) {
            gpuHashDagFilePath = outPathCharArray;
            free(outPathCharArray);
        } else {
            pEngine->destroy();
            return 1;
        }
    }

    CUDA_CHECK_ERROR();

    if (DAG_TYPE == EDag::MyGpuDag && std::filesystem::exists(gpuHashDagFilePath)) {
        PROFILE_SCOPE("load GPU DAG from disk");
        BinaryReader reader { gpuHashDagFilePath };
        pEngine->readFrom(reader);
    } else {
        {
            PROFILE_SCOPE("load colors from disk");
            if (std::filesystem::exists(uncompressedColorsFilePath))
                BasicDAGFactory::load_uncompressed_colors_from_file(pEngine->basicDagUncompressedColors, uncompressedColorsFilePath, EMemoryType::CPU);
            if (std::filesystem::exists(compressedColorsFilePath))
                BasicDAGFactory::load_compressed_colors_from_file(pEngine->basicDagCompressedColors, compressedColorsFilePath, EMemoryType::CPU);
        }

        PROFILE_SCOPE("load DAG from disk");
        BasicDAGFactory::load_dag_from_file(pEngine->dagInfo, pEngine->basicDag, basicDagFilePath, EMemoryType::CPU);

#if 0
        DAGUtils::fix_enclosed_leaves(pEngine->basicDag, pEngine->basicDagCompressedColors.enclosedLeaves, pEngine->basicDagCompressedColors.topLevels);
#if 0
        BasicDAGFactory::save_compressed_colors_to_file(pEngine->basicDagCompressedColors, "data/" FILENAME ".basic_dag.compressed_colors.variable.bin");
        pEngine->basicDagCompressedColors.free();
        BasicDAGFactory::load_compressed_colors_from_file(pEngine->basicDagCompressedColors, "data/" FILENAME ".basic_dag.compressed_colors.variable.bin");
#endif
#endif

        // Determine which voxel textures we want to use for this scene.
        if constexpr (true) {
            SCOPED_STATS("Picking voxel textures");

            std::filesystem::path textureFolder { ROOT_FOLDER };
            textureFolder = textureFolder / "assets" / "textures_phyronnaz_goodvibes";
            if (pEngine->basicDagCompressedColors.is_valid()) {
                const size_t numColors = pEngine->basicDagCompressedColors.get_leaves_count(0, pEngine->basicDag.get_first_node_index());
                pEngine->voxelTextures = VoxelTextures::createSubsetFromFolder(textureFolder, pEngine->basicDagCompressedColors, numColors);
            } else {
                const size_t numColors = pEngine->basicDagUncompressedColors.get_leaves_count(0, pEngine->basicDag.get_first_node_index());
                pEngine->voxelTextures = VoxelTextures::createSubsetFromFolder(textureFolder, pEngine->basicDagUncompressedColors, numColors);
            }
        }

        if (pEngine->basicDagUncompressedColors.is_valid()) {
            if constexpr (DAG_TYPE == EDag::HashDag) {
                HashDAGFactory::load_from_DAG(pEngine->hashDag, pEngine->basicDag, 0x8FFFFFFF / C_pageSize / sizeof(uint32));
            } else if constexpr (DAG_TYPE == EDag::MyGpuDag) {
                MyGPUHashDAGFactory::load_from_DAG(pEngine->myGpuHashDag, pEngine->basicDag, pEngine->basicDagUncompressedColors, pEngine->voxelTextures);
            }
        } else if (pEngine->basicDagCompressedColors.is_valid()) {
            if constexpr (DAG_TYPE == EDag::HashDag) {
                HashDAGFactory::load_from_DAG(pEngine->hashDag, pEngine->basicDag, 0x8FFFFFFF / C_pageSize / sizeof(uint32));
                HashDAGFactory::load_colors_from_DAG(pEngine->hashDagColors, pEngine->basicDag, pEngine->basicDagCompressedColors);
            }
            if constexpr (DAG_TYPE == EDag::MyGpuDag) {
                MyGPUHashDAGFactory::load_from_DAG(pEngine->myGpuHashDag, pEngine->basicDag, pEngine->basicDagCompressedColors, pEngine->voxelTextures);
            }
        }
    }
    CUDA_CHECK_ERROR();

#if defined(SAVE_SCENE) && SAVE_SCENE
    {
        BinaryWriter writer { gpuHashDagFilePath };
        pEngine->writeTo(writer);
    }
#endif // SAVE_SCENE

    if constexpr (DAG_TYPE == EDag::HashDag) {
        pEngine->basicDagColorErrors.uncompressedColors = pEngine->basicDagUncompressedColors;
        pEngine->basicDagColorErrors.compressedColors = pEngine->basicDagCompressedColors;
    } else if constexpr (DAG_TYPE == EDag::BasicDagUncompressedColors) {
        pEngine->basicDag.upload_to_gpu();
        pEngine->basicDagUncompressedColors.upload_to_gpu();
    } else if constexpr (DAG_TYPE == EDag::BasicDagCompressedColors) {
        pEngine->basicDag.upload_to_gpu();
        pEngine->basicDagCompressedColors.upload_to_gpu();
    } else {
        pEngine->basicDagUncompressedColors.free();
        pEngine->basicDagCompressedColors.free();
    }

    pEngine->set_dag(DAG_TYPE);

    CUDA_CHECK_ERROR();

#if USE_VIDEO
    pEngine->toggle_fullscreen();
    pEngine->videoManager.load_video(rootFolder / "videos" / (SCENE "_" VIDEO_NAME ".txt"));
    std::this_thread::sleep_for(std::chrono::seconds(5));
#endif
#if USE_REPLAY
    pEngine->replayReader.load_csv(rootFolder / "replays" / (SCENE "_" REPLAY_NAME ".csv"));
#endif

    printf("Starting...\n");
#ifdef PROFILING_PATH
    if (pEngine->hashDag.is_valid())
        pEngine->hashDag.data.save_bucket_sizes(true);
#endif

#if !defined(SAVE_SCENE) || !SAVE_SCENE
    pEngine->loop();
#endif // SAVE_SCENE

    pEngine->destroy();

    return 0;
}
