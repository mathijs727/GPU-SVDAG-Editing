#pragma once
#include "typedefs.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <memory>
#include <vector>

#include "binary_reader.h"
#include "binary_writer.h"
#include "camera_view.h"
#include "dag_tracer.h"
#include "dags/basic_dag/basic_dag.h"
#include "dags/hash_dag/hash_dag.h"
#include "dags/hash_dag/hash_dag_editors.h"
#include "dags/my_gpu_dags/my_gpu_dag_editors.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag.h"
#include "dags/my_gpu_dags/my_gpu_hash_dag/my_gpu_hash_dag_edits.h"
#include "events.h"
#include "image.h"
#include "opengl_texture.h"
#include "replay.h"
#include "video.h"
#include "voxel_textures.h"
#include <chrono>
#include <concepts>
#include <magic_enum.hpp>
#include <memory>

enum class EDag : int {
    BasicDagUncompressedColors,
    BasicDagCompressedColors,
    BasicDagColorErrors,
    HashDag,
    MyGpuDag,
};

constexpr uint32 CNumDags = (uint32_t)magic_enum::enum_count<EDag>();

// clang-format off
template <typename T>
concept is_gpu_editor = requires(T t) {
    { T::is_gpu_editor };
};
// clang-format on

class Engine : std::enable_shared_from_this<Engine> {
public:
    BasicDAG basicDag;
    HashDAG hashDag;
    MyGPUHashDAG<EMemoryType::GPU_Malloc> myGpuHashDag;
    GpuMemoryPool editingMemPool;
    BasicDAGCompressedColors basicDagCompressedColors;
    BasicDAGUncompressedColors basicDagUncompressedColors;
    BasicDAGColorErrors basicDagColorErrors;
    HashDAGColors hashDagColors;
    HashDAGUndoRedo undoRedo;
    MyGPUHashDAGUndoRedo gpuUndoRedo;
    DAGInfo dagInfo;
    VoxelTextures voxelTextures;
    CameraView view;

    CameraView targetView;
    double targetLerpSpeed = 1;
    CameraView initialView;
    bool moveToTarget = false;
    double targetLerpTime = 0;

    inline void init_target_lerp()
    {
        moveToTarget = true;
        initialView = view;
        targetLerpTime = 0;
    }

    ReplayManager replayReader;
    VideoManager videoManager;

    StatsRecorder statsRecorder;

    struct EditConfig {
        EDag currentDag = EDag::HashDag;
        ETool tool = ETool::Sphere;
        EDebugColors debugColors = EDebugColors::None;
        uint32 debugColorsIndexLevel = 0;
        float radius = 10;
        uint3 copySourcePath = make_uint3(0, 0, 0);
        uint3 copyDestPath = make_uint3(0, 0, 0);
        ToolPath path;

        bool toolSpeedLimited = false;
        float3 editColor = make_float3(0, 0, 1);
        uint32_t editMaterial = 1;
    };
    EditConfig config;
    using clck = std::chrono::high_resolution_clock;
    clck::time_point lastEditTimePoint = clck::now();

    float movementSpeedMultiplier = 1.0f;
    float gravityConstantInVoxels = 0.5f;
    float jumpSpeedInVoxels = 0.16f;

    template <typename T, typename... TArgs>
    void edit(TArgs&&... Args)
    {
        PROFILE_FUNCTION();

        const auto doEditsForDag = [&]() {
            lastEditTimestamp = statsRecorder.get_frame_timestamp();
            lastEditFrame = frameIndex;

            BasicStats stats;
            // is disabled inside the function
            hashDag.data.prefetch();

            stats.start_work("creating edit tool");
            auto tool = T(std::forward<TArgs>(Args)...);
            stats.flush(statsRecorder);

            // Make sure that we don't include GPU rendering time of previous frame in the "total edits" time.
            cudaDeviceSynchronize();
            if constexpr (is_gpu_editor<T>) {
                stats.start_work("total edits");
                editMyHashDag(tool, myGpuHashDag, gpuUndoRedo, statsRecorder, editingMemPool, nullptr);
                cudaDeviceSynchronize();
                stats.flush(statsRecorder);

#if AUTO_GARBAGE_COLLECT
                stats.start_work("garbage_collect");
                gpuUndoRedo.garbageCollect(myGpuHashDag);
                stats.flush(statsRecorder);
#endif
            } else {
                stats.start_work("total edits");
                hashDag.edit_threads(tool, hashDagColors, undoRedo, statsRecorder);
                stats.flush(statsRecorder);

                stats.start_work("upload_to_gpu");
                hashDag.data.upload_to_gpu();
                stats.flush(statsRecorder);

#if AUTO_GARBAGE_COLLECT
                stats.start_work("garbage_collect");
                hashDag.remove_stale_nodes(hashDag.levels - 2);
                undoRedo.free();
                stats.flush(statsRecorder);
#endif
            }

            statsRecorder.reportFloat("radius", tool.radius, "voxels", Device::CPU); // Voxels or world units?
            statsRecorder.printLastFrame();
        };

        // Only run the code once to prevent duplicate stats (total edits & upload_to_gpu)
        if constexpr (!is_gpu_editor<T>) {
            if (config.currentDag == EDag::HashDag)
                doEditsForDag();
        } else {
            if (config.currentDag == EDag::MyGpuDag)
                doEditsForDag();
        }
    }
    void set_dag(EDag dag);

    void init(bool headLess);
    void loop();
    void destroy();

    void toggle_fullscreen();

    static std::unique_ptr<Engine> create();

    void readFrom(BinaryReader& reader);
    void writeTo(BinaryWriter& reader) const;

private:
    Engine() = default;

    struct InputState {
        std::vector<bool> keys = std::vector<bool>(GLFW_KEY_LAST + 1, false);
        std::vector<bool> mouse = std::vector<bool>(8, false);
        double mousePosX = 0;
        double mousePosY = 0;
    };

    GLFWwindow* window = nullptr;
    GLuint image = 0;

    InputState state;

    GLuint programID = 0;
    GLint textureID = 0;
    GLuint fsvao = 0;

    bool replayStarted = true;
    double dt = 0;
    bool headLess = false;
    bool firstReplay = true;
    bool printMemoryStats = false;
    bool shadows = true;
    int ambientOcclusionSamplesPerPixel = 3;
    bool pathTracing = true;
    bool pathTracingIntegratePixel = true;
    int pathTracingSamplesPerPixel = 1;
#ifdef DEFAULT_PATH_TRACE_DEPTH
    int maxPathTracingDepth = DEFAULT_PATH_TRACE_DEPTH;
#else
    int maxPathTracingDepth = 4;
#endif
    float environmentBrightness = 1.0f;
    float shadowBias = 1;
    float aoRayLength = 1 << (SCENE_DEPTH - 10);
    float fogDensity = 0;
    bool showUI = true;
    float swirlPeriod = 100;
    bool enableSwirl = true;
    bool creativeMode = true;
    double verticalSpeedInVoxels = 0.0;
    bool fullscreen = false;
    bool vsync = false;
    Vector3 transformRotation = { 0, 0, 0 };
    float transformScale = 1;
    std::chrono::high_resolution_clock::time_point time;
    uint32 frameIndex = 0;
    std::unique_ptr<EventsManager> eventsManager;
    std::unique_ptr<DAGTracer> tracer;
    ReplayManager replayWriter;

    struct Timings {
        double pathsTime = 0;
        double colorsTime = 0;
        double shadowsTime = 0;
        double ambientOcclusionTime = 0;
        double ambientOcclusionBlurTime = 0;
        double lightingTime = 0;
        double pathTracingTime = 0;
        double totalTimeGPU = 0;
        double totalTimeCPU = 0;
    };
    Timings timings;

    uint32 lastEditTimestamp = 0;
    uint32 lastEditFrame = 0;

    bool is_dag_valid(EDag dag) const;
    void next_dag();
    void previous_dag();

    static void key_callback(GLFWwindow*, int key, int scancode, int action, int mods);
    static void mouse_callback(GLFWwindow*, int button, int action, int mods);
    static void scroll_callback(GLFWwindow*, double xoffset, double yoffset);

    void key_callback_impl(int key, int scancode, int action, int mods);
    void mouse_callback_impl(int button, int action, int mods);
    void scroll_callback_impl(double xoffset, double yoffset);

    void tick();

    void init_graphics();

    void loop_headless();
    void loop_graphics();
};
