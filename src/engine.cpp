#include <fmt/format.h>
#include <fmt/ostream.h>
// Include fmt before typedefs.h
#include "configuration/profile_definitions.h"
#include "dags/hash_dag/hash_dag_editors.h"
#include "dags/my_gpu_dags/my_gpu_dag_editors.h"
#include "engine.h"
#include "events.h"
#include "hacky_profiler.hpp"
#include "memory.h"
#include "my_units.h"
#include "my_units_fmt.h"
#include "shader.h"
#include "typedefs.h"
#include "utils.h"
#include <cmath> // std::lerp
#include <filesystem>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <imgui_stdlib.h>
#include <magic_enum.hpp>
#include <nfd.h>
#define FREEIMAGE_LIB
#include <FreeImage.h>

static const auto rootFolder = std::filesystem::path(ROOT_FOLDER);
const auto bricksFolder = rootFolder / "assets" / "heightmaps" / "bricks";
const auto bricksHeightMapFile = bricksFolder / "brick_wall2-disp-512.png";
const auto bricksColorMapFile = bricksFolder / "brick_wall2-diff-512.tga";

static float3 paintColor = make_float3(0, 1, 0);

// #undef EXIT_AFTER_REPLAY
// #define EXIT_AFTER_REPLAY 1

inline void clear_console()
{
    fmt::print("\033[H\033[J");
}

bool Engine::is_dag_valid(EDag dag) const
{
    const auto isValid = [](auto dag, auto colors) {
#if EDITS_ENABLE_COLORS
        return dag.is_valid() && colors.is_valid();
#else
        return dag.is_valid();
#endif
    };
    switch (dag) {
    case EDag::BasicDagUncompressedColors:
        return isValid(basicDag, basicDagUncompressedColors);
    case EDag::BasicDagCompressedColors:
        return isValid(basicDag, basicDagCompressedColors);
    case EDag::BasicDagColorErrors:
        return isValid(basicDag, basicDagColorErrors);
    case EDag::HashDag:
        return isValid(hashDag, hashDagColors);
    case EDag::MyGpuDag:
        return myGpuHashDag.is_valid();
    default:
        check(false);
        return false;
    }
}

void Engine::next_dag()
{
    do {
        config.currentDag = EDag((uint32(config.currentDag) + 1) % CNumDags);
    } while (!is_dag_valid(config.currentDag));
}

void Engine::previous_dag()
{
    do {
        config.currentDag = EDag(Utils::subtract_mod(uint32(config.currentDag), CNumDags));
    } while (!is_dag_valid(config.currentDag));
}

void Engine::set_dag(EDag dag)
{
    config.currentDag = dag;
    if (!is_dag_valid(config.currentDag)) {
        next_dag();
    }
}

void Engine::key_callback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
    auto* pEngine = (Engine*)glfwGetWindowUserPointer(pWindow);
    pEngine->key_callback_impl(key, scancode, action, mods);
}
void Engine::mouse_callback(GLFWwindow* pWindow, int button, int action, int mods)
{
    auto* pEngine = (Engine*)glfwGetWindowUserPointer(pWindow);
    pEngine->mouse_callback_impl(button, action, mods);
}
void Engine::scroll_callback(GLFWwindow* pWindow, double xoffset, double yoffset)
{
    auto* pEngine = (Engine*)glfwGetWindowUserPointer(pWindow);
    pEngine->scroll_callback_impl(xoffset, yoffset);
}

void Engine::key_callback_impl(int key, int scancode, int action, int mods)
{
    // Ignore keyboard inputs when user is interacting with imgui
    // https://github.com/ocornut/imgui/blob/master/docs/FAQ.md#q-how-can-i-tell-whether-to-dispatch-mousekeyboard-to-dear-imgui-or-my-application
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureKeyboard)
        return;

    if (!((0 <= key) && (key <= GLFW_KEY_LAST))) // Media keys
    {
        return;
    }

    if (action == GLFW_RELEASE)
        state.keys[(uint64)key] = false;
    if (action == GLFW_PRESS)
        state.keys[(uint64)key] = true;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_B) {
            replayStarted = true;
        }
        if (key == GLFW_KEY_M) {
            printMemoryStats = !printMemoryStats;
        }
#if UNDO_REDO
        if (key == GLFW_KEY_Z) {
            if (config.currentDag == EDag::HashDag || config.currentDag == EDag::MyGpuDag) {
                if (state.keys[GLFW_KEY_LEFT_CONTROL] || state.keys[GLFW_KEY_RIGHT_CONTROL]) {
                    if (state.keys[GLFW_KEY_LEFT_SHIFT]) {
                        if (hashDag.is_valid())
                            undoRedo.redo(hashDag, hashDagColors);
                        if (myGpuHashDag.is_valid())
                            gpuUndoRedo.redo(myGpuHashDag);
                        replayWriter.add_action<ReplayActionRedo>();
                    } else {
                        if (hashDag.is_valid())
                            undoRedo.undo(hashDag, hashDagColors);
                        if (myGpuHashDag.is_valid())
                            gpuUndoRedo.undo(myGpuHashDag);
                        replayWriter.add_action<ReplayActionUndo>();
                    }
                }
            }
        }
#endif
        if (key == GLFW_KEY_BACKSPACE) {
            replayWriter.write_csv();
            replayWriter.clear();
            fmt::print("Replay saved!\n");
        }
        if (key == GLFW_KEY_R) {
            if (state.keys[GLFW_KEY_LEFT_SHIFT]) {
                fmt::print("Replay reader cleared\n");
                fmt::print("Replay writer cleared\n");
                replayReader.clear();
                replayWriter.clear();
            } else {
                fmt::print("Replay reader reset\n");
                fmt::print("Stats cleared\n");
                statsRecorder.clear();
                replayReader.reset_replay();
            }
        }
        if (key == GLFW_KEY_TAB) {
            if (state.keys[GLFW_KEY_LEFT_SHIFT]) {
                config.tool = ETool(Utils::subtract_mod(uint32(config.tool), CNumTools));
            } else {
                config.tool = ETool((uint32(config.tool) + 1) % CNumTools);
            }

            fmt::print("Current tool: {}\n", magic_enum::enum_name(config.tool));
        }
        if (key == GLFW_KEY_G) {
            if (config.currentDag == EDag::HashDag) {
                hashDag.remove_stale_nodes(hashDag.levels - 2);
            }
            undoRedo.free();
        }
        if (key == GLFW_KEY_C) {
            if (state.keys[GLFW_KEY_LEFT_SHIFT]) {
                config.debugColors = EDebugColors(
                    Utils::subtract_mod(uint32(config.debugColors), CNumDebugColors));
            } else {
                config.debugColors = EDebugColors((uint32(config.debugColors) + 1) % CNumDebugColors);
            }
        }
        if (key == GLFW_KEY_U) {
            auto previousGPUUsage = Memory::get_gpu_allocated_memory();
            auto previousCPUUsage = Memory::get_cpu_allocated_memory();
            undoRedo.free();
            fmt::print(
                "Undo redo cleared! Memory saved: GPU: {}MB CPU: {}MB\n",
                Utils::to_MB(previousGPUUsage - Memory::get_gpu_allocated_memory()),
                Utils::to_MB(previousCPUUsage - Memory::get_cpu_allocated_memory()));
        }
        if (key == GLFW_KEY_CAPS_LOCK) {
            if (state.keys[GLFW_KEY_LEFT_SHIFT]) {
                previous_dag();
            } else {
                next_dag();
            }
            fmt::print("Current dag: {}\n", magic_enum::enum_name(config.currentDag));
        }
        if (key == GLFW_KEY_1) {
            config.debugColorsIndexLevel++;
            config.debugColorsIndexLevel = std::min(config.debugColorsIndexLevel, basicDag.levels);
        }
        if (key == GLFW_KEY_2) {
            config.debugColorsIndexLevel = uint32(std::max(int32(config.debugColorsIndexLevel) - 1, 0));
        }
        if (key == GLFW_KEY_3) {
            config.debugColors = EDebugColors::Index;
        }
        if (key == GLFW_KEY_4) {
            config.debugColors = EDebugColors::Position;
        }
        if (key == GLFW_KEY_5) {
            config.debugColors = EDebugColors::ColorTree;
        }
        if (key == GLFW_KEY_6) {
            config.debugColors = EDebugColors::ColorBits;
        }
        if (key == GLFW_KEY_7) {
            config.debugColors = EDebugColors::MinColor;
        }
        if (key == GLFW_KEY_8) {
            config.debugColors = EDebugColors::MaxColor;
        }
        if (key == GLFW_KEY_9) {
            config.debugColors = EDebugColors::Weight;
        }
        if (key == GLFW_KEY_0) {
            config.debugColors = EDebugColors::None;
        }
        if (key == GLFW_KEY_X) {
            shadows = !shadows;
        }
        if (key == GLFW_KEY_EQUAL) {
            shadowBias += 0.1f;
            fmt::print("Shadow bias: {}\n", shadowBias);
        }
        if (key == GLFW_KEY_MINUS) {
            shadowBias -= 0.1f;
            fmt::print("Shadow bias: {}\n", shadowBias);
        }
        if (key == GLFW_KEY_O) {
            fogDensity += 1;
            fmt::print("Fog density: {}\n", fogDensity);
        }
        if (key == GLFW_KEY_H) {
            showUI = !showUI;
        }
        if (key == GLFW_KEY_F) {
            toggle_fullscreen();
        }

        const double rotationStep = (state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT]
                ? -10
                : 10);
        if (key == GLFW_KEY_F1) {
            transformRotation.X += rotationStep;
            if (transformRotation.X > 180)
                transformRotation.X -= 360;
            if (transformRotation.X < -180)
                transformRotation.X += 360;
        }
        if (key == GLFW_KEY_F2) {
            transformRotation.Y += rotationStep;
            if (transformRotation.Y > 180)
                transformRotation.Y -= 360;
            if (transformRotation.Y < -180)
                transformRotation.Y += 360;
        }
        if (key == GLFW_KEY_F3) {
            transformRotation.Z += rotationStep;
            if (transformRotation.Z > 180)
                transformRotation.Z -= 360;
            if (transformRotation.Z < -180)
                transformRotation.Z += 360;
        }
        if (key == GLFW_KEY_F6) {
            transformScale += state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT]
                ? -.1f
                : .1f;
        }

        if (key == GLFW_KEY_F4) {
            enableSwirl = !enableSwirl;
        }
        if (key == GLFW_KEY_F5) {
            swirlPeriod += state.keys[GLFW_KEY_LEFT_SHIFT] || state.keys[GLFW_KEY_RIGHT_SHIFT]
                ? -10.0f
                : 10.0f;
        }

        if (key == GLFW_KEY_I) {
            fogDensity -= 1;
            fmt::print("Fog density: {}\n", fogDensity);
        }
        if (key == GLFW_KEY_P) {
            const bool printGlobalStats = state.keys[GLFW_KEY_LEFT_SHIFT];
            if (config.currentDag == EDag::BasicDagUncompressedColors) {
                if (printGlobalStats)
                    DAGUtils::print_stats(basicDag);
                basicDag.print_stats();
                basicDagUncompressedColors.print_stats();
            } else if (config.currentDag == EDag::BasicDagCompressedColors) {
                if (printGlobalStats)
                    DAGUtils::print_stats(basicDag);
                basicDag.print_stats();
                basicDagCompressedColors.print_stats();
            } else if (config.currentDag == EDag::BasicDagColorErrors) {
                if (printGlobalStats)
                    DAGUtils::print_stats(basicDag);
                basicDag.print_stats();
            } else if (config.currentDag == EDag::HashDag) {
                if (printGlobalStats)
                    DAGUtils::print_stats(hashDag);
                hashDag.data.print_stats();
                hashDagColors.print_stats();
#if UNDO_REDO
                undoRedo.print_stats();
#endif
            } else {
                fmt::print("TODO");
                hashDagColors.print_stats();
#if UNDO_REDO
                undoRedo.print_stats();
#endif
            }
        }
        if (key == GLFW_KEY_L && hashDag.is_valid()) {
            hashDag.data.save_bucket_sizes(false);
        }
        if (key == GLFW_KEY_KP_ENTER) {
            fmt::print("view.rotation = {{ {}, {}, {}, {}, {}, {}, {}, {}, {} }};\n",
                view.rotation.D00, view.rotation.D01, view.rotation.D02,
                view.rotation.D10, view.rotation.D11, view.rotation.D12,
                view.rotation.D20, view.rotation.D21, view.rotation.D22);
            fmt::print("view.position = {{ {}, {}, {} }};\n", view.position.X,
                view.position.Y, view.position.Z);
        }
    }
}

void Engine::mouse_callback_impl(int button, int action, int mods)
{
    // Ignore mouse inputs when user is interacting with imgui
    // https://github.com/ocornut/imgui/blob/master/docs/FAQ.md#q-how-can-i-tell-whether-to-dispatch-mousekeyboard-to-dear-imgui-or-my-application
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    if (button != GLFW_MOUSE_BUTTON_LEFT && button != GLFW_MOUSE_BUTTON_RIGHT)
        return;

    if (action == GLFW_RELEASE) {
        state.mouse[(uint64)button] = false;
    } else if (action == GLFW_PRESS) {
        state.mouse[(uint64)button] = true;
    }
}

void Engine::scroll_callback_impl(double xoffset, double yoffset)
{
    // Ignore mouse inputs when user is interacting with imgui
    // https://github.com/ocornut/imgui/blob/master/docs/FAQ.md#q-how-can-i-tell-whether-to-dispatch-mousekeyboard-to-dear-imgui-or-my-application
    const ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse)
        return;

    config.radius += float(yoffset) * (state.keys[GLFW_KEY_LEFT_SHIFT] ? 10.f : 1.f);
    config.radius = std::max(config.radius, 0.f);
}

void Engine::tick()
{
    PROFILE_FUNCTION();
    CUDA_CHECK_ERROR();

    frameIndex++;

    BasicStats stats;
    stats.start_work("frame");

    if (printMemoryStats) {
        clear_console();
        std::cout << Memory::get_stats_string();
    }

    videoManager.tick(*this);

    std::function<uint32_t(int, int, int)> v = [](int, int, int) { return 0; };
    const double3 boundsMin = make_double3(dagInfo.boundsAABBMin);
    const double3 boundsMax = make_double3(dagInfo.boundsAABBMax);
    const double3 scale = make_double3(double(1 << MAX_LEVELS)) / (boundsMax - boundsMin);
    if (!creativeMode) {
        const double3 translation = -boundsMin;
        const double3 cameraPosition = make_double3(view.position);
        const float3 finalCameraPosition = make_float3((cameraPosition + translation) * scale);

        const int3 playerCenter = make_int3((unsigned)finalCameraPosition.x, (unsigned)finalCameraPosition.y - 1u, (unsigned)finalCameraPosition.z);

        static constexpr int clipMapSize = 2;
        static constexpr int clipMapExtent = 1 + clipMapSize * 2;
        const auto voxelIndex = [](int dx, int dy, int dz) {
            const int x = dx + clipMapSize;
            const int y = dy + clipMapSize;
            const int z = dz + clipMapSize;
            return z * clipMapExtent * clipMapExtent + y * clipMapExtent + x;
        };
        std::vector<uint3> playerClipMap((size_t)(clipMapExtent * clipMapExtent * clipMapExtent));
        for (int dz = -clipMapSize; dz <= clipMapSize; ++dz) {
            for (int dy = -clipMapSize; dy <= clipMapSize; ++dy) {
                for (int dx = -clipMapSize; dx <= clipMapSize; ++dx) {
                    playerClipMap[voxelIndex(dx, dy, dz)] = make_uint3(playerCenter + make_int3(dx, dy, dz));
                }
            }
        }
        std::vector<uint32_t> voxels(playerClipMap.size());
        tracer->get_voxel_values(myGpuHashDag, playerClipMap, voxels);
        v = [=](int dx, int dy, int dz) { return voxels[voxelIndex(dx, dy, dz)] != (uint32_t)-1; };
    }
    const auto canMoveInDirection = [&](const Vector3& vdir) {
        if (vdir.X >= 0.0 && vdir.Z >= 0.0)
            return !v(1, 0, 0) && !v(1, 0, 1) && !v(0, 0, 1);
        if (vdir.X >= 0.0 && vdir.Z < 0.0)
            return !v(1, 0, 0) && !v(1, 0, -1) && !v(0, 0, -1);
        if (vdir.X < 0.0 && vdir.Z >= 0.0)
            return !v(-1, 0, 0) && !v(-1, 0, 1) && !v(0, 0, 1);
        if (vdir.X < 0.0 && vdir.Z < 0.0)
            return !v(-1, 0, 0) && !v(-1, 0, -1) && !v(0, 0, -1);
        return true;
    };

    // Controls
    if (replayReader.is_empty() || replayReader.at_end()) {
        if (state.keys[GLFW_KEY_KP_0]) {
            targetView.rotation = { -0.573465, 0.000000, -0.819230,
                -0.034067, 0.999135, 0.023847,
                0.818522, 0.041585, -0.572969 };
            targetView.position = { -13076.174715, -1671.669438, 5849.331627 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_1]) {
            targetView.rotation = { 0.615306, -0.000000, -0.788288,
                -0.022851, 0.999580, -0.017837,
                0.787957, 0.028989, 0.615048 };
            targetView.position = { -7736.138941, -2552.420373, -5340.566371 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_2]) {
            targetView.rotation = { -0.236573, -0.000000, -0.971614,
                0.025623, 0.999652, -0.006239,
                0.971276, -0.026372, -0.236491 };
            targetView.position = { -2954.821641, 191.883613, 4200.793442 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_3]) {
            targetView.rotation = { 0.590287, -0.000000, -0.807193, 0.150128, 0.982552,
                0.109786, 0.793109, -0.185987, 0.579988 };
            targetView.position = { -7036.452685, -3990.109906, 7964.129876 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_4]) {
            targetView.rotation = { -0.222343, -0.000000, -0.974968,
                0.070352, 0.997393, -0.016044,
                0.972427, -0.072159, -0.221764 };
            targetView.position = { 762.379376, -935.456405, -358.642203 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_5]) {
            targetView.rotation = { 0, 0, -1, 0, 1, 0, 1, 0, 0 };
            targetView.position = { -951.243605, 667.199855, -27.706481 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_6]) {
            targetView.rotation = { 0.095015, -0.000000, -0.995476, 0.130796, 0.991331,
                0.012484, 0.986846, -0.131390, 0.094192 };
            targetView.position = { 652.972238, 73.188250, -209.028828 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_7]) {
            targetView.rotation = { -0.004716, -0.000000, -0.999989,
                0.583523, 0.812093, -0.002752,
                0.812084, -0.583529, -0.003830 };
            targetView.position = { -1261.247484, 1834.904220, -11.976059 };
            init_target_lerp();
        }
        if (state.keys[GLFW_KEY_KP_8]) {
            targetView.rotation = { 0.019229, -0.000000, 0.999815,
                -0.040020, 0.999198, 0.000770,
                -0.999014, -0.040027, 0.019213 };
            targetView.position = { -8998.476232, -2530.419704, -4905.593975 };
            init_target_lerp();
        }

        if (state.keys[GLFW_KEY_KP_ADD])
            config.radius++;

        const double voxelToWorld = 1.0 / scale.x;
        double speed = creativeMode ? length(make_double3(dagInfo.boundsAABBMax - dagInfo.boundsAABBMin)) / 100 * dt * movementSpeedMultiplier : std::min(dt * movementSpeedMultiplier * 10, 0.9999) * voxelToWorld;
        double rotationSpeed = 2 * dt;

        if (state.keys[GLFW_KEY_LEFT_SHIFT])
            speed *= 10;

        // In gravity mode you should not be moving up/down when looking up/down.
        auto forward = view.forward();
        forward = Vector3::Normalized(forward * Vector3(1, 0, 1));
        if (state.keys[GLFW_KEY_W] && canMoveInDirection(forward)) {
            view.position += speed * forward;
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_S] && canMoveInDirection(-forward)) {
            view.position -= speed * forward;
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_D] && canMoveInDirection(view.right())) {
            view.position += speed * view.right();
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_A] && canMoveInDirection(-view.right())) {
            view.position -= speed * view.right();
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_SPACE]) {
            if (creativeMode) {
                view.position += speed * view.up();
                moveToTarget = false;
            } else {
                if (verticalSpeedInVoxels == 0)
                    verticalSpeedInVoxels = (double)jumpSpeedInVoxels;
            }
        }
        if (state.keys[GLFW_KEY_LEFT_CONTROL] && creativeMode) {
            view.position -= speed * view.up();
            moveToTarget = false;
        }

        if (state.keys[GLFW_KEY_RIGHT] || state.keys[GLFW_KEY_E]) {
            view.rotation *= Matrix3x3::FromQuaternion(
                Quaternion::FromAngleAxis(rotationSpeed, Vector3::Up()));
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_LEFT] || state.keys[GLFW_KEY_Q]) {
            view.rotation *= Matrix3x3::FromQuaternion(
                Quaternion::FromAngleAxis(-rotationSpeed, Vector3::Up()));
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_DOWN]) {
            view.rotation *= Matrix3x3::FromQuaternion(
                Quaternion::FromAngleAxis(-rotationSpeed, view.right()));
            moveToTarget = false;
        }
        if (state.keys[GLFW_KEY_UP]) {
            view.rotation *= Matrix3x3::FromQuaternion(
                Quaternion::FromAngleAxis(rotationSpeed, view.right()));
            moveToTarget = false;
        }

        if (!creativeMode) {
            const double gravityInVoxels = (double)gravityConstantInVoxels * dt;
            verticalSpeedInVoxels -= gravityInVoxels;
            if (v(0, -1, 0) != 0)
                verticalSpeedInVoxels = std::max(verticalSpeedInVoxels, 0.0);
            verticalSpeedInVoxels = std::clamp(verticalSpeedInVoxels, -0.9, 0.9); // Clamp to prevent user from falling through the floor at too high speed.
            view.position.Y += verticalSpeedInVoxels * voxelToWorld;
        }
    }

    if (moveToTarget) {
        targetLerpTime = clamp(targetLerpTime + targetLerpSpeed * dt, 0., 1.);
        view.position = lerp(initialView.position, targetView.position, targetLerpTime);
        view.rotation = Matrix3x3::FromQuaternion(Quaternion::Slerp(
            Matrix3x3::ToQuaternion(initialView.rotation),
            Matrix3x3::ToQuaternion(targetView.rotation), targetLerpTime));
    }

    if (replayReader.is_empty()) {
        // Save position/rotation
        replayWriter.add_action<ReplayActionSetLocation>(view.position);
        replayWriter.add_action<ReplayActionSetRotation>(view.rotation);
    } else if (!replayReader.at_end() && replayStarted) {
        replayReader.replay_frame();
        if (replayReader.at_end()) {
            if (firstReplay && REPLAY_TWICE) {
                fmt::print(
                    "First replay ended, starting again now that everything is loaded "
                    "in memory...\n");
                firstReplay = false;
                replayReader.reset_replay();
                statsRecorder.clear();
            } else {
                // #if OPTIMIZE_FOR_BENCHMARK
                fmt::print("Replay ended, saving stats... ");
                statsRecorder.write_csv();
                statsRecorder.write_json();
// #endif
#ifdef PROFILING_PATH
                if (config.currentDag == EDag::HashDag)
                    hashDag.data.save_bucket_sizes(false);
#endif
                statsRecorder.clear();
                fmt::print("Saved!\n");
            }
        }
    }

    // Trace primary visibility.
    std::chrono::duration<double> pathsTime { 0 };
    switch (config.currentDag) {
    case EDag::BasicDagUncompressedColors:
    case EDag::BasicDagCompressedColors:
    case EDag::BasicDagColorErrors:
        pathsTime = tracer->resolve_paths(view, dagInfo, basicDag);
        break;
    case EDag::HashDag:
        pathsTime = tracer->resolve_paths(view, dagInfo, hashDag);
        break;
    case EDag::MyGpuDag:
        pathsTime = tracer->resolve_paths(view, dagInfo, myGpuHashDag);
        break;
    }
    statsRecorder.report("paths", pathsTime, Device::GPU);
    CUDA_CHECK_ERROR();

    // Compute the 3D location of the users cursor.
    constexpr double xMultiplier = double(imageWidth) / windowWidth;
    constexpr double yMultiplier = double(imageHeight) / windowHeight;
    const uint32 posX = uint32(
        clamp<int32>(int32(xMultiplier * state.mousePosX), 0, imageWidth - 1));
    const uint32 posY = uint32(
        clamp<int32>(int32(yMultiplier * state.mousePosY), 0, imageHeight - 1));
    if (replayReader.is_empty()) {
#if !OPTIMIZE_FOR_BENCHMARK
        config.path = tracer->get_path(make_uint2(posX, posY));
#else
        (void)posX;
        (void)posY;
#endif

#if RECORD_TOOL_OVERLAY
        replayWriter.add_action<ReplayActionSetToolParameters>(
            config.path, config.copySourcePath, config.copyDestPath, config.radius,
            uint32(config.tool));
#endif
    }

    // Compute material color at the voxels scene by the camera.
    // cudaDeviceSynchronize();
    using Clock = std::chrono::high_resolution_clock;
    static auto prevFrame = Clock::now();
    const auto curFrame = Clock::now();
    prevFrame = curFrame;
    const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(curFrame - prevFrame);
    constexpr std::chrono::milliseconds targetFrameTime { 250 };
    if (delta > targetFrameTime)
        printf("WARNING: FRAMERATE TOO LOW\n");
    // else
    //     std::this_thread::sleep_for(targetFrameTime - delta);
    std::chrono::duration<double> colorsTime { 0 };
    const uint32 debugColorsIndexLevel = basicDag.levels - 2 - config.debugColorsIndexLevel;
    const ToolInfo toolInfo { config.tool, config.path, std::max(config.radius, 1.0f), config.copySourcePath, config.copyDestPath };
    switch (config.currentDag) {
    case EDag::BasicDagUncompressedColors:
        colorsTime = tracer->resolve_colors(basicDag, basicDagUncompressedColors,
            voxelTextures, config.debugColors, debugColorsIndexLevel);
        break;
    case EDag::BasicDagCompressedColors:
        colorsTime = tracer->resolve_colors(basicDag, basicDagCompressedColors,
            voxelTextures, config.debugColors, debugColorsIndexLevel);
        break;
    case EDag::BasicDagColorErrors:
        colorsTime = tracer->resolve_colors(basicDag, basicDagColorErrors,
            voxelTextures, config.debugColors, debugColorsIndexLevel);
        break;
    case EDag::HashDag:
        colorsTime = tracer->resolve_colors(hashDag, hashDagColors,
            voxelTextures, config.debugColors, debugColorsIndexLevel);
        break;
    case EDag::MyGpuDag:
        colorsTime = tracer->resolve_colors(myGpuHashDag, hashDagColors,
            voxelTextures, config.debugColors, debugColorsIndexLevel);
        break;
    default: {
    } break;
    }
    statsRecorder.report("colors", colorsTime, Device::GPU);
    CUDA_CHECK_ERROR();

    // Timing variables.
    std::chrono::duration<double> shadowsTime { 0 };
    std::chrono::duration<double> ambientOcclusionTime { 0 };
    std::chrono::duration<double> ambientOcclusionBlurTime { 0 };
    std::chrono::duration<double> lightingTime { 0 };
    std::chrono::duration<double> pathTracingTime { 0 };

    if (pathTracing) {
        // Compute path traced lighting.
        const auto runPathTracing = [&](const auto& dag, const auto& colors) {
#if !OPTIMIZE_FOR_BENCHMARK
            (void)posX;
            (void)posY;
            config.path = tracer->get_path(make_uint2(posX, posY));
#endif
            colorsTime = tracer->resolve_colors(dag, colors, voxelTextures, config.debugColors, debugColorsIndexLevel);
            pathTracingTime = tracer->resolve_path_tracing(
                view, dagInfo, dag, colors,
                voxelTextures, config.debugColors, debugColorsIndexLevel, toolInfo,
                environmentBrightness, pathTracingSamplesPerPixel, maxPathTracingDepth, pathTracingIntegratePixel);
        };

        switch (config.currentDag) {
        case EDag::BasicDagUncompressedColors: {
            runPathTracing(basicDag, basicDagUncompressedColors);
        } break;
        case EDag::BasicDagCompressedColors: {
            runPathTracing(basicDag, basicDagCompressedColors);
        } break;
        case EDag::BasicDagColorErrors: {
            runPathTracing(basicDag, basicDagColorErrors);
        } break;
        case EDag::HashDag: {
            runPathTracing(hashDag, hashDagColors);
        } break;
        case EDag::MyGpuDag: {
            runPathTracing(myGpuHashDag, hashDagColors); // Colors are unused...
        } break;
        }
        statsRecorder.report("path_tracing", pathTracingTime, Device::GPU);
        CUDA_CHECK_ERROR();
    } else {
        // Compute direct lighting with optional shadows and ray-traced ambient occlusion.
        if constexpr (ENABLE_SHADOWS) {
            const auto computeLighting = [&](auto dag) {
                if (ambientOcclusionSamplesPerPixel > 0) {
                    ambientOcclusionTime = tracer->resolve_ambient_occlusion(dag, ambientOcclusionSamplesPerPixel, shadowBias, aoRayLength);
                    CUDA_CHECK_ERROR();
                    ambientOcclusionBlurTime = tracer->resolve_ambient_occlusion_blur();
                    CUDA_CHECK_ERROR();
                }
                if (shadows) {
                    shadowsTime = tracer->resolve_shadows(dag, shadowBias);
                    CUDA_CHECK_ERROR();
                }
            };
            switch (config.currentDag) {
            case EDag::BasicDagUncompressedColors:
            case EDag::BasicDagCompressedColors:
            case EDag::BasicDagColorErrors:
                computeLighting(basicDag);
                break;
            case EDag::HashDag:
                computeLighting(hashDag);
                break;
            case EDag::MyGpuDag:
                computeLighting(myGpuHashDag);
                break;
            };
        }
        statsRecorder.report("shadows", shadowsTime, Device::GPU);
        statsRecorder.report("ambient_occlusion", ambientOcclusionTime, Device::GPU);
        statsRecorder.report("ambient_occlusion_blur", ambientOcclusionBlurTime, Device::GPU);

        lightingTime = tracer->resolve_lighting(view, dagInfo, toolInfo, ambientOcclusionSamplesPerPixel > 0, shadows, fogDensity);
        statsRecorder.report("lighting", lightingTime, Device::GPU);
        CUDA_CHECK_ERROR();
    }

    auto currentTime = std::chrono::high_resolution_clock::now();
    const double historyWeight = 0.95;
    timings.pathsTime = std::lerp(Utils::to_ms(pathsTime), timings.pathsTime, historyWeight);
    timings.colorsTime = std::lerp(Utils::to_ms(colorsTime), timings.colorsTime, historyWeight);
    timings.shadowsTime = std::lerp(Utils::to_ms(shadowsTime), timings.shadowsTime, historyWeight);
    timings.ambientOcclusionTime = std::lerp(Utils::to_ms(ambientOcclusionTime), timings.ambientOcclusionTime, historyWeight);
    timings.ambientOcclusionBlurTime = std::lerp(Utils::to_ms(ambientOcclusionBlurTime), timings.ambientOcclusionBlurTime, historyWeight);
    timings.lightingTime = std::lerp(Utils::to_ms(lightingTime), timings.lightingTime, historyWeight);
    timings.pathTracingTime = std::lerp(Utils::to_ms(pathTracingTime), timings.pathTracingTime, historyWeight);
    timings.totalTimeGPU = std::lerp(Utils::to_ms(pathsTime + colorsTime + shadowsTime + ambientOcclusionTime + ambientOcclusionBlurTime + lightingTime + pathTracingTime), timings.totalTimeGPU, historyWeight);
    timings.totalTimeCPU = std::lerp(Utils::to_ms(currentTime - time), timings.totalTimeCPU, historyWeight);
    dt = std::chrono::duration<double>(currentTime - time).count();
    time = currentTime;

    if ((config.currentDag == EDag::HashDag || config.currentDag == EDag::MyGpuDag) && replayReader.is_empty()) {
        const float toolRateLimitInMs = config.toolSpeedLimited ? 200.0f : 0.0f;
        const auto timeSinceLastEditInMs = std::chrono::duration<float, std::milli>(clck::now() - lastEditTimePoint).count();

        if ((state.mouse[GLFW_MOUSE_BUTTON_LEFT] || state.mouse[GLFW_MOUSE_BUTTON_RIGHT]) && timeSinceLastEditInMs >= toolRateLimitInMs) {
            lastEditTimePoint = clck::now();

            if (config.tool == ETool::CubeCopy && state.mouse[GLFW_MOUSE_BUTTON_RIGHT]) {
                if (state.keys[GLFW_KEY_LEFT_SHIFT]) {
                    config.copySourcePath = config.path.centerPath;
                } else {
                    config.copyDestPath = config.path.centerPath;
                }
            }

            const bool isAdding = state.mouse[GLFW_MOUSE_BUTTON_RIGHT];
            const float3 centerPosition = make_float3(config.path.centerPath);
            const float3 neighbourPosition = make_float3(config.path.neighbourPath);

            if (config.tool == ETool::Sphere) {
                // printf("position = (%f, %f, %f); radius = %f\n", position.x, position.y, position.z, config.radius);
                if (isAdding) {
                    edit<SphereEditor<true>>(neighbourPosition, config.radius);
                    edit<MyGpuSphereEditor<true>>(neighbourPosition, config.radius, config.editMaterial);
                    replayWriter.add_action<ReplayActionSphere>(neighbourPosition, config.radius, isAdding, config.editMaterial);
                } else {
                    edit<SphereEditor<false>>(centerPosition, config.radius);
                    edit<MyGpuSphereEditor<false>>(centerPosition, config.radius, config.editMaterial);
                    replayWriter.add_action<ReplayActionSphere>(centerPosition, config.radius, isAdding, config.editMaterial);
                }
            } else if (config.tool == ETool::SpherePaint) {
                edit<SpherePaintEditor>(centerPosition, config.radius);
                edit<MyGpuSpherePaintEditor>(centerPosition, config.radius, paintColor, config.editMaterial);
                replayWriter.add_action<ReplayActionPaint>(centerPosition, config.radius, paintColor, config.editMaterial);
            } else if (config.tool == ETool::SphereNoise) {
                edit<SphereNoiseEditor>(hashDag, centerPosition, config.radius, isAdding);
            } else if (config.tool == ETool::Cube) {
                if (isAdding) {
                    edit<BoxEditor<true>>(neighbourPosition, config.radius);
                    edit<MyGpuBoxEditor<true>>(neighbourPosition, config.radius, config.editMaterial);
                    replayWriter.add_action<ReplayActionCube>(neighbourPosition, config.radius, isAdding, config.editMaterial);
                } else {
                    edit<BoxEditor<false>>(centerPosition, config.radius);
                    edit<MyGpuBoxEditor<false>>(centerPosition, config.radius, config.editMaterial);
                    replayWriter.add_action<ReplayActionCube>(centerPosition, config.radius, isAdding, config.editMaterial);
                }
            } else if (config.tool == ETool::CubeCopy) {
                if (!isAdding && config.radius >= 1) {
                    const float3 src = make_float3(config.copySourcePath);
                    const float3 dest = make_float3(config.copyDestPath);
                    const Matrix3x3 transform = Matrix3x3::FromQuaternion(
                                                    Quaternion::FromEuler(transformRotation / 180 * M_PI))
                        * transformScale;
                    edit<CopyEditor>(hashDag, hashDagColors, src, dest, centerPosition, config.radius, transform, statsRecorder, enableSwirl, swirlPeriod);
                    edit<MyGpuCopyEditor<MyGPUHashDAG<EMemoryType::GPU_Malloc>>>(myGpuHashDag, src, dest, centerPosition, config.radius);
                    replayWriter.add_action<ReplayActionCopy>(centerPosition, src, dest,
                        config.radius, transform,
                        enableSwirl, swirlPeriod);
                }
            } else if (config.tool == ETool::CubeFill) {
                const float3 center = centerPosition + (isAdding ? -1.f : 1.f) * round(2.f * make_float3(view.forward()));
                edit<FillEditorColors>(hashDag, hashDagColors, center, config.radius);
                replayWriter.add_action<ReplayActionFill>(center, config.radius);
            }
        }
    }

    stats.flush(statsRecorder);
    CUDA_CHECK_ERROR();

    if (replayReader.is_empty()) {
        PROFILE_SCOPE("addReplay");
        replayWriter.add_action<ReplayActionEndFrame>();
    } else {
        PROFILE_SCOPE("report");
        if (hashDag.data.is_valid()) {
            statsRecorder.report("virtual_size",
                hashDag.data.get_virtual_used_size(false), Device::CPU);
            statsRecorder.report("allocated_size",
                hashDag.data.get_allocated_pages_size(), Device::CPU);
            statsRecorder.report("color_size", hashDagColors.memory_used(), Device::CPU);
            statsRecorder.report("color_size undo_redo",
                undoRedo.memory_used(), Device::CPU);
#if SIMULATE_GC
            hashDag.simulate_remove_stale_nodes(statsRecorder);
#endif
        } else if (myGpuHashDag.is_valid()) {
            myGpuHashDag.report(statsRecorder);
        }
    }
    statsRecorder.next_frame();

    HACK_PROFILE_FRAME_ADVANCE();
}

void Engine::init(bool inheadLess)
{
    PROFILE_FUNCTION();

    FreeImage_Initialise();

    headLess = inheadLess;

    if (!headLess) {
        init_graphics();
    }

    // std::filesystem::path textureFolder { ROOT_FOLDER };
    // textureFolder = textureFolder / "assets" / "textures";
    // voxelTextures = VoxelTextures::createFromStructuredFolder(textureFolder);

    eventsManager = std::make_unique<EventsManager>();
    tracer = std::make_unique<DAGTracer>(headLess, eventsManager.get());
    image = tracer->get_colors_image();
    time = std::chrono::high_resolution_clock::now();

    // cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 1024 * 8);
    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256 * 1024 * 1024);
    editingMemPool = GpuMemoryPool::create(nullptr);
}

void Engine::init_graphics()
{
    PROFILE_FUNCTION();

    // Initialize GLFW
    if (!glfwInit()) {
        const char* description;
        int code = glfwGetError(&description);
        fmt::print(stderr, "Failed to initialize GLFW\n{} - {}", code, description);
        exit(1);
    }

    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
        GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
    // Open a window and create its OpenGL context
    window = glfwCreateWindow(windowWidth, windowHeight, "DAG Edits", NULL, NULL);

    if (window == NULL) {
        fmt::print(stderr,
            "Failed to open GLFW window. If you have an Intel GPU, they are "
            "not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
        glfwTerminate();
        exit(1);
    }
    glfwMakeContextCurrent(window);
    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSwapInterval(vsync ? 1 : 0);

    // Create ImGui context
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // Initialize GLEW
    glewExperimental = true; // Needed for core profile
    if (glewInit() != GLEW_OK) {
        fmt::print(stderr, "Failed to initialize GLEW\n");
        exit(1);
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Dark blue background
    glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    glGenVertexArrays(1, &fsvao);
    glBindVertexArray(fsvao);

    // Create and compile our GLSL program from the shaders
    programID = LoadShaders(rootFolder / "src/shaders/TransformVertexShader.glsl",
        rootFolder / "src/shaders/TextureFragmentShader.glsl");

    // Get a handle for our "myTextureSampler" uniform
    textureID = glGetUniformLocation(programID, "myTextureSampler");

    // Our vertices. Tree consecutive floats give a 3D vertex; Three consecutive
    // vertices give a triangle. A cube has 6 faces with 2 triangles each, so this
    // makes 6*2=12 triangles, and 12*3 vertices
    static const GLfloat g_vertex_buffer_data[] = {
        -1.0f,
        -1.0f,
        0.0f,
        1.0f,
        -1.0f,
        0.0f,
        1.0f,
        1.0f,
        0.0f,
        -1.0f,
        -1.0f,
        0.0f,
        1.0f,
        1.0f,
        0.0f,
        -1.0f,
        1.0f,
        0.0f,
    };

    // Two UV coordinates for each vertex. They were created with Blender.
    static const GLfloat g_uv_buffer_data[] = {
        0.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };

    GLuint vertexBuffer = 0;
    glGenBuffers(1, &vertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data),
        g_vertex_buffer_data, GL_STATIC_DRAW);

    GLuint uvBuffer = 0;
    glGenBuffers(1, &uvBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data), g_uv_buffer_data,
        GL_STATIC_DRAW);

    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glVertexAttribPointer(0, // attribute. No particular reason for 0, but must
                             // match the layout in the shader.
        3, // size
        GL_FLOAT, // type
        GL_FALSE, // normalized?
        0, // stride
        (void*)0 // array buffer offset
    );

    // 2nd attribute buffer : UVs

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
    glVertexAttribPointer(1, // attribute. No particular reason for 1, but must
                             // match the layout in the shader.
        2, // size : U+V => 2
        GL_FLOAT, // type
        GL_FALSE, // normalized?
        0, // stride
        (void*)0 // array buffer offset
    );

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDeleteBuffers(1, &vertexBuffer);
    glDeleteBuffers(1, &uvBuffer);
}

void Engine::loop()
{
    if (headLess) {
        loop_headless();
    } else {
        loop_graphics();
    }
}

void Engine::loop_headless()
{
    PROFILE_FUNCTION();

    while (!replayReader.at_end()) {
        MARK_FRAME();
        tick();
    }
}

template <typename T>
constexpr auto magic_enum_names()
{
    constexpr auto names = magic_enum::enum_names<T>();
    std::array<const char*, names.size()> out;
    for (size_t i = 0; i < names.size(); i++)
        out[i] = names[i].data();
    return out;
}

void Engine::loop_graphics()
{
    CUDA_CHECK_ERROR();
    PROFILE_FUNCTION();

    do {
        MARK_FRAME();

        glfwGetCursorPos(window, &state.mousePosX, &state.mousePosY);

        // Without calls to cudaDeviceSynchronize(...) we may overfeed the GPU with commands (if we submit faster than it can process).
        // The CUDA / OpenGL interop doesn't fix this because they are asynchronous and CUDA does not seem to synchronize with OpenGL vsync.
        // This function manually ensures that we can have at most 2 CUDA frames in flight at any time.
        // eventsManager->forceFrameSync(0);

        tick();

        // Copy the frame buffer from CUDA to OpenGL
        tracer->update_colors_image();

        glfwSetWindowTitle(window, "HashDag");

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use our shader
        glUseProgram(programID);

        // Send our transformation to the currently bound shader,
        // in the "MVP" uniform
        // glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

        // Bind our texture in Texture Unit 0
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, image);

        // Set our "myTextureSampler" sampler to use Texture Unit 0
        glUniform1i(textureID, 0);

        // Draw the triangle !
        glBindVertexArray(fsvao);
        glDrawArrays(GL_TRIANGLES, 0,
            12 * 3); // 12*3 indices starting at 0 -> 12 triangles

        glBindVertexArray(0);
        glUseProgram(0);

        // 2D stuff
        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (showUI) {
            PROFILE_SCOPE("ImGui");

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // ImGui::SetNextWindowCollapsed(false);
            if (ImGui::Begin("Main Window")) {
                ImGui::Text("Cursor: %u, %u, %u", config.path.centerPath.x, config.path.centerPath.y, config.path.centerPath.z);
                ImGui::Checkbox("Creative Mode", &creativeMode);

                ImGui::Spacing();
                ImGui::SliderFloat("Movement Speed", &movementSpeedMultiplier, 0.05f, 20.0f);
                if (!creativeMode) {
                    ImGui::SliderFloat("Gravity", &gravityConstantInVoxels, 0.1f, 3.0f);
                    ImGui::SliderFloat("Jump Speed", &jumpSpeedInVoxels, 0.1f, 3.0f);
                }
                ImGui::Spacing();

                if (ImGui::Button("Save")) {
                    nfdchar_t* outPathCharArray = nullptr;
                    nfdresult_t result = NFD_SaveDialog(nullptr, nullptr, &outPathCharArray);

                    if (result == NFD_OKAY) {
                        std::string outPathStr { outPathCharArray };
                        outPathStr += std::to_string(1 << (SCENE_DEPTH - 10)) + "k";
                        outPathStr += ".gpu_hash_dag.dag.bin";
                        free(outPathCharArray);

                        cudaDeviceSynchronize();
                        BinaryWriter writer { outPathStr };
                        writer.write(*this);
                    }
                }

                if (ImGui::TreeNode("Help")) {
                    const auto addKey = [](const char* key, const char* action) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(key);
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(action);
                    };
                    ImGui::Text("Tool:");
                    ImGui::BeginTable("Tool", 2);
                    addKey("[LMSB]", "delete / copy");
                    addKey("[RMSB]", "fill / paint / set copy dest");
                    addKey("[RMSB+SHIFT]", "set copy source");
                    addKey("[SCROLL]", "increase / decrease tool size");
                    ImGui::EndTable();
                    ImGui::Text("Look:");
                    ImGui::BeginTable("Look Table", 2);
                    addKey("[Q]", "look left");
                    addKey("[E]", "look right");
                    ImGui::EndTable();
                    ImGui::Spacing();
                    ImGui::Text("Movement:");
                    ImGui::BeginTable("Movement Table", 2);
                    addKey("[W]", "move forward");
                    addKey("[A]", "move left");
                    addKey("[S]", "move back");
                    addKey("[D]", "move right");
                    addKey("[CTRL]", "move down");
                    addKey("[SPACE]", "move up");
                    addKey("[SHIFT]", "move faster");
                    ImGui::EndTable();
                    ImGui::TreePop();
                }
                ImGui::Spacing();
                if (ImGui::TreeNodeEx("Render Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                    static const auto debugColors = magic_enum_names<EDebugColors>();
                    ImGui::Combo("Debug Colors", reinterpret_cast<int*>(&config.debugColors), debugColors.data(), (int)debugColors.size());
                    ImGui::Checkbox("Path Tracing", &pathTracing);
                    if (pathTracing) {
                        ImGui::Checkbox("Integrate Pixel Domain", &pathTracingIntegratePixel);
                        ImGui::Text("Accumulated Samples: %i", tracer->get_path_tracing_num_accumulated_samples());
                        ImGui::InputInt("Samples Per Pixel", &pathTracingSamplesPerPixel, 1);
                        pathTracingSamplesPerPixel = std::max(pathTracingSamplesPerPixel, 0);
                        ImGui::InputInt("Max Path Depth", &maxPathTracingDepth, 1);
                        ImGui::InputFloat("Brightness", &environmentBrightness);
                    } else {
                        ImGui::Checkbox("Shadows", &shadows);
                        ImGui::InputFloat("Shadow bias", &shadowBias, 0.1f);
                        ImGui::InputInt("Num AO Samples", &ambientOcclusionSamplesPerPixel, 1);
                        ambientOcclusionSamplesPerPixel = std::max(ambientOcclusionSamplesPerPixel, 0);
                        ImGui::InputFloat("AO ray length", &aoRayLength, 10.0f);
                        ImGui::InputFloat("Fog density", &fogDensity, 1.0f);
                    }
                    if (ImGui::Button("Toggle fullscreen"))
                        toggle_fullscreen();
                    if (ImGui::Checkbox("Vsync", &vsync)) {
                        glfwSwapInterval(vsync ? 1 : 0);
                    }
                    ImGui::TreePop();
                }
                ImGui::Spacing();
                if (ImGui::TreeNodeEx("Tools", ImGuiTreeNodeFlags_DefaultOpen)) {
                    static const auto dags = magic_enum_names<EDag>();
                    if (ImGui::Combo("DAG type", reinterpret_cast<int*>(&config.currentDag), dags.data(), (int)dags.size()))
                        set_dag(config.currentDag);
                    static const auto tools = magic_enum_names<ETool>();
                    ImGui::Combo("Current Tool", reinterpret_cast<int*>(&config.tool), tools.data(), (int)tools.size());
                    ImGui::Text("Tool radius: %f", config.radius);
                    ImGui::Checkbox("Limit tool speed", &config.toolSpeedLimited);

                    if (config.tool == ETool::Cube || config.tool == ETool::Sphere || config.tool == ETool::SpherePaint) {
                        const auto getName = [](void* pData, int idx, const char** ppOutText) -> bool {
                            const auto* pThis = (const Engine*)(pData);
                            if (idx < 0 || idx >= (int)pThis->voxelTextures.materialNames.size())
                                return false;

                            *ppOutText = pThis->voxelTextures.materialNames[idx].c_str();
                            return true;
                        };
                        int signedEditMaterial = config.editMaterial;
                        ImGui::Combo("Material", &signedEditMaterial, getName, (void*)this, (int)voxelTextures.materialNames.size());
                        config.editMaterial = (uint32_t)signedEditMaterial;
                        // ImGui::ColorPicker3("Color", (float*)&paintColor);
                    }

                    ImGui::Spacing();

                    ImGui::BeginTable("Undo/Redo Table", 2);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    if (ImGui::Button("<< Undo")) {
                        undoRedo.undo(hashDag, hashDagColors);
                        gpuUndoRedo.undo(myGpuHashDag);
                        replayWriter.add_action<ReplayActionUndo>();
                    }
                    ImGui::TableNextColumn();
                    if (ImGui::Button("Redo >>")) {
                        undoRedo.redo(hashDag, hashDagColors);
                        gpuUndoRedo.redo(myGpuHashDag);
                        replayWriter.add_action<ReplayActionRedo>();
                    }
                    ImGui::EndTable();
                    if (ImGui::Button("Clear Undo/Redo")) {
                        auto previousGPUUsage = Memory::get_gpu_allocated_memory();
                        auto previousCPUUsage = Memory::get_cpu_allocated_memory();
                        undoRedo.free();
                        fmt::print(
                            "Undo redo cleared! Memory saved: GPU: {}MB CPU: {}MB\n",
                            Utils::to_MB(previousGPUUsage - Memory::get_gpu_allocated_memory()),
                            Utils::to_MB(previousCPUUsage - Memory::get_cpu_allocated_memory()));
                    }

                    ImGui::Spacing();

                    if (ImGui::Button("Save replay")) {
                        replayWriter.write_csv();
                        replayWriter.clear();
                        fmt::print("Replay saved!");
                    }
                    if (ImGui::Button("Clear replay reader/writer")) {
                        fmt::print("Replay reader cleared\n");
                        fmt::print("Replay writer cleared\n");
                        replayReader.clear();
                        replayWriter.clear();
                    }
                    if (ImGui::Button("Reset replay reader")) {
                        fmt::print("Replay reader reset\n");
                        fmt::print("Stats cleared\n");
                        statsRecorder.clear();
                        replayReader.reset_replay();
                    }
                    ImGui::TreePop();
                }
                ImGui::Spacing();
                if (ImGui::TreeNodeEx("Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
#define STRINGIFY0_(x) #x
#define STRINGIFY_(x) STRINGIFY0_(x)
                    ImGui::Text("Scene %s (2^%d) using %s", STRINGIFY_(SCENE), SCENE_DEPTH, magic_enum::enum_name(config.currentDag).data());
                    ImGui::Text("Active tool: %s", magic_enum::enum_name(config.tool).data());

                    if (config.tool == ETool::CubeCopy) {
#if COPY_APPLY_TRANSFORM
                        F(hx, y, EFmt::glow, "Rotation:", dx, vector3_(transformRotation));
                        y -= 24.f;
                        F(hx, y, EFmt::glow, "Scale:", dx, std::to_string(transformScale));
                        y -= 24.f;
#endif
#if COPY_CAN_APPLY_SWIRL
                        F(hx, y, EFmt::glow, "Swirl:", dx, enableSwirl ? "ON" : "OFF");
                        y -= 24.f;
                        F(hx, y, EFmt::glow, "Swirl period:", dx,
                            std::to_string(swirlPeriod));
                        y -= 24.f;
#endif
                    }

                    const double editingAndUploading = lastEditFrame == frameIndex
                        ? statsRecorder.get_value_in_frame(lastEditTimestamp,
                              "total edits")
                            + statsRecorder.get_value_in_frame(lastEditTimestamp,
                                "upload_to_gpu")
                            + statsRecorder.get_value_in_frame(lastEditTimestamp,
                                "creating edit tool")
                        : 0;

                    if (ImGui::TreeNodeEx("Timings", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::BeginTable("Timings Table", 2);
                        const auto drawTiming = [](const char* name, double timing) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::TextUnformatted(name);
                            ImGui::TableNextColumn();
                            const std::string str = fmt::format("{:6.2f}ms", timing); // Using FMT::format here gives issues with custom new/delete
                            ImGui::TextUnformatted(str.c_str());
                        };
                        drawTiming("Trace paths", timings.pathsTime);
                        drawTiming("Resolve colors", timings.colorsTime);
                        drawTiming("Trace shadow", timings.shadowsTime);
                        drawTiming("Trace AO", timings.ambientOcclusionTime);
                        drawTiming("Blur AO", timings.ambientOcclusionBlurTime);
                        drawTiming("Lighting", timings.lightingTime);
                        drawTiming("Path Tracing", timings.pathTracingTime);
                        drawTiming("Total GPU (approximate)", timings.totalTimeGPU);
                        drawTiming("Edit & Upload", editingAndUploading);
                        drawTiming("Total CPU", timings.totalTimeCPU);
                        ImGui::EndTable();
                        ImGui::TreePop();
                    }

#if !OPTIMIZE_FOR_BENCHMARK
                    if (ImGui::TreeNodeEx("Memory", ImGuiTreeNodeFlags_DefaultOpen)) {
                        // Using FMT::format here gives issues with custom new/delete
                        auto const mb_ = [&](auto T) {
                            // return fmt::format("{:6.1f}MB", value);
                            return fmt::format("{:6.3f}MB", Utils::to_MB(T));
                        };
                        auto const cmb_ = [&](auto T, auto U) {
                            // return fmt::format("{} ({:6.1f}MB)", T, U);
                            return fmt::format("{} ({})", T, U);
                        };
                        auto const mbx_ = [&](auto T, auto U) {
                            // return fmt::format("{:6.1f}MB (+{:6.1f}MB)", T, U);
                            return fmt::format("{} (+{})", T, U);
                        };
                        [[maybe_unused]] auto const mb2_ = [&](auto T, auto U) {
                            // return fmt::format("{:6.1f}MB / {6.1f}MB", T, U);
                            return fmt::format("{:6.1f}MB / {:6.1f}MB", Utils::to_MB(T), Utils::to_MB(U));
                        };

                        const auto drawMemory = [&](const char* name, std::string memoryStr) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::TextUnformatted(name);
                            ImGui::TableNextColumn();
                            ImGui::TextUnformatted(memoryStr.c_str());
                        };
                        if (ImGui::TreeNodeEx("BasicDAG", ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::BeginTable("BasicDAG Memory Table", 2);
                            drawMemory("BasicDAG", mb_(basicDag.memory_used()));
                            drawMemory("BasicDAG Uncompressed Colors", mb_(basicDagUncompressedColors.memory_used()));
                            drawMemory("BasicDAG Compressed Colors", mb_(basicDagCompressedColors.memory_used()));
                            ImGui::EndTable();
                            ImGui::TreePop();
                        }
                        if (ImGui::TreeNodeEx("HashDAG", ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::BeginTable("HashDAG Memory Table", 2);
                            drawMemory("HashDAG Page pool", cmb_(hashDag.data.get_total_pages(), hashDag.data.get_pool_size()));
                            drawMemory("HashDAG Used", cmb_(hashDag.data.get_allocated_pages(), hashDag.data.get_allocated_pages_size()));
                            drawMemory("HashDAG Page table", mb_(hashDag.data.get_page_table_size()));
                            drawMemory("HashDAG Colors", mb_(hashDagColors.memory_used()));
                            ImGui::EndTable();
                            ImGui::TreePop();
                        }
                        if (ImGui::TreeNodeEx("GPUHashDAG", ImGuiTreeNodeFlags_DefaultOpen)) {
                            ImGui::BeginTable("GPUHashDAG Memory Table", 2);
                            drawMemory("GPUHashDAG (items used)", mb_(myGpuHashDag.memory_used_by_items()));
                            drawMemory("GPUHashDAG (slabs used)", mb_(myGpuHashDag.memory_used_by_slabs()));
                            drawMemory("GPUHashDAG (allocated)", mb_(myGpuHashDag.memory_allocated()));
                            ImGui::EndTable();

                            if (ImGui::Button("Garbage Collect")) {
                                gpuUndoRedo.garbageCollect(myGpuHashDag);
                            }
                            ImGui::TreePop();
                        }

                        ImGui::BeginTable("Total Memory Table", 2);
#if USE_VIDEO
                        drawMemory("Total (GPU/CPU)", mb2_(Utils::to_MB(Memory::get_gpu_allocated_memory()), Utils::to_MB(Memory::get_cpu_allocated_memory() + Memory::get_cxx_cpu_allocated_memory())));
#else
                        drawMemory("Total GPU (allocated) ", mb_(Utils::to_MB(Memory::get_gpu_allocated_memory())));
                        drawMemory("Total GPU (used)      ", mb_(Utils::to_MB(Memory::get_gpu_used_memory())));
                        drawMemory("Total GPU (peak)      ", mb_(Utils::to_MB(Memory::get_gpu_peak_allocated_memory())));
                        drawMemory("Total CPU             ", mbx_(Utils::to_MB(Memory::get_cpu_allocated_memory()), Utils::to_MB(Memory::get_cxx_cpu_allocated_memory())));
#endif
                        ImGui::EndTable();

                        ImGui::Checkbox("Print memory stats", &printMemoryStats);
                        if (ImGui::Button("Garbage Collection")) {
                            if (config.currentDag == EDag::HashDag) {
                                hashDag.remove_stale_nodes(hashDag.levels - 2);
                            }
                            undoRedo.free();
                        }
                        ImGui::TreePop();
                    }
#endif

                    if (ImGui::TreeNodeEx("Edits", ImGuiTreeNodeFlags_DefaultOpen)) {
                        ImGui::BeginTable("Edits Table", 2);
                        const auto drawEdit = [&](const char* name, double count) {
                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            ImGui::TextUnformatted(name);
                            ImGui::TableNextColumn();
                            ImGui::Text("%4.3e", count);
                        };
                        drawEdit("Num Voxels", statsRecorder.get_value_in_frame(lastEditTimestamp, "num voxels"));
                        drawEdit("Num Nodes", statsRecorder.get_value_in_frame(lastEditTimestamp, "num nodes"));
                        ImGui::EndTable();
                        ImGui::TreePop();
                    }
                    ImGui::TreePop();
                }

                ImGui::End();
            }

            ImGui::EndFrame();
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);

        {
            PROFILE_SCOPE("glfwSwapBuffers");
            // Swap buffers
            glfwSwapBuffers(window);
        }
        glfwPollEvents();
    } // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0
#if EXIT_AFTER_REPLAY
        && !replayReader.at_end()
#endif
    );
}

void Engine::destroy()
{
    // Wait for GPU to finish processing before we delete any resources.
    cudaDeviceSynchronize();

    FreeImage_DeInitialise();

    if (fsvao != 0)
        glDeleteVertexArrays(1, &fsvao);

    tracer.reset();
    basicDag.free();
    basicDagCompressedColors.free();
    basicDagUncompressedColors.free();
    basicDagColorErrors.free();
    hashDagColors.free();
    hashDag.free();
    myGpuHashDag.free();
    undoRedo.free();
    voxelTextures.free();
    editingMemPool.release();
}

void Engine::toggle_fullscreen()
{
    if (!fullscreen) {
        fullscreen = true;
        GLFWmonitor* primary = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(primary);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0, mode->width,
            mode->height, mode->refreshRate);
    } else {
        fullscreen = false;
        glfwSetWindowMonitor(window, NULL, 0, 0, windowWidth, windowHeight, -1);
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
}

std::unique_ptr<Engine> Engine::create()
{
    auto pEngine = std::unique_ptr<Engine>(new Engine()); // Don't use make_unique(Engine {}) because that calls move (which I'm too lazy to implement).
    pEngine->replayReader = ReplayManager(pEngine.get());
    pEngine->replayWriter = ReplayManager(pEngine.get());
    return pEngine;
}

void Engine::readFrom(BinaryReader& reader)
{
    reader.read(dagInfo);
    reader.read(myGpuHashDag);
    reader.read(view);
    reader.read(voxelTextures);
}

void Engine::writeTo(BinaryWriter& writer) const
{
    writer.write(dagInfo);
    writer.write(myGpuHashDag);
    writer.write(view);
    writer.write(voxelTextures);
}
