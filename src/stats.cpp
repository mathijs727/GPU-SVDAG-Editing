#include "typedefs.h"
//
#include <nlohmann/json.hpp>
// Include first because we get some issues otherwise (on Windows with MSVC)
#include "configuration/gpu_hash_dag_definitions.h"
#include "configuration/profile_definitions.h"
#include "engine.h" // EDAG::...
#include "my_units_fmt.h"
#include "stats.h"
#include "utils.h"
#include <algorithm>
#include <array>
#include <date/date.h>
#if ENABLE_GIT_COMMIT_IN_STATS
#include <git.h> // https://github.com/andrew-hardin/cmake-git-version-tracking
#endif
#include <iomanip>
#include <magic_enum.hpp>
#include <string>

#ifdef WIN32
#pragma push_macro("FORCEINLINE")
#undef FORCEINLINE
#include <Windows.h>
#undef FORCEINLINE
#pragma pop_macro("FORCEINLINE")
#else
#include <unistd.h> // gethostname
#endif

std::vector<Stats*> Stats::stack;

std::string getHostNameCpp()
{
#ifdef WIN32
    std::array<char, MAX_COMPUTERNAME_LENGTH + 1> hostName;
    DWORD size = (DWORD)hostName.size();
    GetComputerNameA(hostName.data(), &size);
    return std::string(hostName.data(), size);
#else
    std::array<char, 1024> hostName;
    gethostname(hostName.data(), hostName.size());
    return std::string(hostName.data());
#endif
}

std::string getOperatingSystemName()
{
#ifdef WIN32
    return "Windows";
#else
    return "Unix";
#endif
}

StatsRecorder::StatsRecorder()
{
    next_frame();
}

void StatsRecorder::next_frame()
{
    frames.emplace_back();
}

void StatsRecorder::report(std::string name, const std::chrono::duration<double>& duration, Device device)
{
    frames.back().elements.push_back(Element {
        .device = device,
        .name = std::move(name),
        .value = duration });
}

void StatsRecorder::report(std::string name, const my_units::bytes storageAmount, Device device)
{
    frames.back().elements.push_back(Element {
        .device = device,
        .name = std::move(name),
        .value = storageAmount });
}

void StatsRecorder::reportFloat(std::string name, double value, std::string unit, Device device)
{
    frames.back().elements.push_back(Element {
        .device = device,
        .name = std::move(name),
        .value = CustomUnit<double> { unit, value } });
}

void StatsRecorder::reportInt(std::string name, int64_t value, std::string unit, Device device)
{
    frames.back().elements.push_back(Element {
        .device = device,
        .name = std::move(name),
        .value = CustomUnit<int64_t> { unit, value } });
}

void StatsRecorder::clear()
{
    frames.clear();
    next_frame();
}

void StatsRecorder::write_csv(std::ostream& stream)
{
    stream << "frame,device,name,time\n";
    for (uint32_t frameIdx = 0; frameIdx < frames.size() - 1; ++frameIdx) { // Last frame might be incomplete.
        const auto& frame = frames[frameIdx];
        for (auto& element : frame.elements) {
            stream << frameIdx;
            stream << ",";
            stream << magic_enum::enum_name(element.device);
            stream << ",";
            stream << element.name;
            stream << ",";

            std::visit(
                Utils::visitor {
                    [&](const std::chrono::duration<double>& duration) {
                        auto durationInMs = std::chrono::duration<double, std::milli>(duration);
                        stream << durationInMs.count();
                    },
                    [&](const my_units::bytes& bytes) {
                        stream << fmt::format("{}", bytes);
                    },
                    [&](const CustomUnit<double>& customUnit) {
                        stream << customUnit.value << " " << customUnit.unit;
                    },
                    [&](const CustomUnit<int64_t>& customUnit) {
                        stream << customUnit.value << " " << customUnit.unit;
                    } },
                element.value);
            stream << "\n";
        }
    }
}

void StatsRecorder::write_csv()
{
    std::filesystem::path fileBasePath;
    std::string fileName;
#if defined(PROFILING_PATH)
    fileBasePath = PROFILING_PATH;
    fileName = STATS_FILES_PREFIX;
#else
    fileBasePath = "./profiling/";
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::stringstream fileNameBuilder;
    fileNameBuilder << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    fileName = fileNameBuilder.str();
#endif
    fileName += ".stats.csv";
    if (!std::filesystem::exists(fileBasePath))
        std::filesystem::create_directories(fileBasePath);

    const auto filePath = fileBasePath / fileName;
    std::ofstream os(filePath);
    checkAlways(os.is_open());

    write_csv(os);

    checkAlways(os.good());
    os.close();
}

void StatsRecorder::write_json()
{
    std::filesystem::path fileBasePath;
    std::string fileName;
#ifdef PROFILING_PATH
    fileBasePath = PROFILING_PATH;
    fileName = STATS_FILES_PREFIX;
#else
    fileBasePath = "./profiling/";
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::stringstream fileNameBuilder;
    fileNameBuilder << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    fileName = fileNameBuilder.str();
#endif
    fileName += ".stats.json";
    if (!std::filesystem::exists(fileBasePath))
        std::filesystem::create_directories(fileBasePath);

    const auto filePath = fileBasePath / fileName;
    if (std::filesystem::exists(filePath))
        std::filesystem::remove(filePath);
    std::ofstream os(filePath);
    checkAlways(os.is_open());

    write_json(os);

    checkAlways(os.good());
    os.close();
}

nlohmann::json getSystemInfoJson()
{
    nlohmann::json out;
    out["hostname"] = getHostNameCpp();
    out["operating_system"] = getOperatingSystemName();
    out["timestamp"] = date::format("%D %T %Z", std::chrono::system_clock::now());
#if DENABLE_GIT_COMMIT_IN_STATS
    out["git_commit_sha1"] = git_CommitSHA1();
    out["git_commit_date"] = git_CommitDate();
#endif
    return out;
}

nlohmann::json getDefineInfoJson()
{
    nlohmann::json hashDagDefinitions {};
    hashDagDefinitions["hash_table_type"] = magic_enum::enum_name(HASH_TABLE_TYPE);
    hashDagDefinitions["target_load_factor"] = TARGET_LOAD_FACTOR;
    hashDagDefinitions["hash_table_warp_add"] = HASH_TABLE_WARP_ADD;
    hashDagDefinitions["hash_table_warp_find"] = HASH_TABLE_WARP_FIND;
    hashDagDefinitions["hash_table_accurate_reserve_memory"] = HASH_TABLE_ACCURATE_RESERVE_MEMORY;
    hashDagDefinitions["hash_table_hash_method"] = magic_enum::enum_name(HASH_TABLE_HASH_METHOD);
    hashDagDefinitions["hash_table_store_slabs_in_table"] = HASH_TABLE_STORE_SLABS_IN_TABLE;
    hashDagDefinitions["hash_dag_material_bits"] = HASH_DAG_MATERIAL_BITS;

    nlohmann::json scriptDefinitions {};
    scriptDefinitions["scene"] = SCENE;
    scriptDefinitions["scene_depth"] = SCENE_DEPTH;
    scriptDefinitions["use_replay"] = USE_REPLAY;
    scriptDefinitions["replay_name"] = REPLAY_NAME;
    scriptDefinitions["replay_depth"] = REPLAY_DEPTH;
    scriptDefinitions["dag_type"] = magic_enum::enum_name<DAG_TYPE>();
    scriptDefinitions["use_bloom_filter"] = USE_BLOOM_FILTER;
    scriptDefinitions["threaded_edits"] = THREADED_EDITS;
    scriptDefinitions["edits_counters"] = EDITS_COUNTERS;
    scriptDefinitions["copy_apply_transform"] = COPY_APPLY_TRANSFORM;
    scriptDefinitions["copy_can_apply_swirl"] = COPY_CAN_APPLY_SWIRL;
    scriptDefinitions["verbose_edit_times"] = VERBOSE_EDIT_TIMES;
    scriptDefinitions["copy_without_decompression"] = COPY_WITHOUT_DECOMPRESSION;
    scriptDefinitions["edits_enable_colors"] = EDITS_ENABLE_COLORS;
    scriptDefinitions["edits_enable_materials"] = EDITS_ENABLE_MATERIALS;
    scriptDefinitions["auto_garbage_collect"] = AUTO_GARBAGE_COLLECT;
    scriptDefinitions["enable_checks"] = ENABLE_CHECKS;
    scriptDefinitions["track_global_newdelete"] = TRACK_GLOBAL_NEWDELETE;
    scriptDefinitions["undo_redo"] = UNDO_REDO;
    scriptDefinitions["color_tree_levels"] = COLOR_TREE_LEVELS;
    scriptDefinitions["edit_parallel_tree_levels"] = EDIT_PARALLEL_TREE_LEVELS;
    scriptDefinitions["capture_gpu_timings"] = CAPTURE_GPU_TIMINGS;
    scriptDefinitions["capture_memory_stats_slow"] = CAPTURE_MEMORY_STATS_SLOW;
    scriptDefinitions["optimize_for_benchmark"] = OPTIMIZE_FOR_BENCHMARK;
    scriptDefinitions["exit_after_replay"] = EXIT_AFTER_REPLAY;
#ifdef DEFAULT_PATH_TRACE_DEPTH
    scriptDefinitions["DEFAULT_PATH_TRACE_DEPTH"] = DEFAULT_PATH_TRACE_DEPTH;
#else
    scriptDefinitions["DEFAULT_PATH_TRACE_DEPTH"] = 0;
#endif

#ifdef ENABLE_VULKAN_MEMORY_ALLOCATOR
    scriptDefinitions["use_vulkan_memory_allocator"] = 1;
#else
    scriptDefinitions["use_vulkan_memory_allocator"] = 0;
#endif

    nlohmann::json out;
    out["script_definitions"] = scriptDefinitions;
    out["gpu_hash_dag_definitions"] = hashDagDefinitions;
    return out;
}

void StatsRecorder::write_json(std::ostream& stream)
{
    std::vector<nlohmann::json> jsonFrames;
    for (const auto& frame : frames) {
        std::vector<nlohmann::json> jsonElements;
        for (const auto& element : frame.elements) {
            nlohmann::json jsonElement;
            jsonElement["device"] = magic_enum::enum_name(element.device);
            jsonElement["name"] = element.name;
            std::visit(
                Utils::visitor {
                    [&](const std::chrono::duration<double>& duration) {
                        jsonElement["unit"] = "seconds";
                        jsonElement["value"] = duration.count();
                    },
                    [&](const my_units::bytes& bytes) {
                        jsonElement["unit"] = "bytes";
                        // jsonElement["value"] = bytes.value();
                        jsonElement["value"] = bytes;
                    },
                    [&](const CustomUnit<double>& customUnit) {
                        jsonElement["unit"] = customUnit.unit;
                        jsonElement["value"] = customUnit.value;
                    },
                    [&](const CustomUnit<int64_t>& customUnit) {
                        jsonElement["unit"] = customUnit.unit;
                        jsonElement["value"] = customUnit.value;
                    } },
                element.value);
            jsonElements.push_back(jsonElement);
        }

        nlohmann::json jsonFrame {};
        jsonFrame["frame"] = jsonFrames.size();
        jsonFrame["stats"] = jsonElements;
        jsonFrames.push_back(jsonFrame);
    }

    nlohmann::json out = getDefineInfoJson();
    out["stats"] = jsonFrames;
    out["machine"] = getSystemInfoJson();
    stream << std::setfill(' ') << std::setw(4) << out;
}

void StatsRecorder::printLastFrame() const
{
    if (frames.empty())
        return;

    const auto& frame = frames.back();
    for (const auto& element : frame.elements) {
        const std::string deviceStr { magic_enum::enum_name(element.device) };
        printf("%s [%s] ", element.name.c_str(), deviceStr.c_str());
        std::visit(
            Utils::visitor {
                [&](const std::chrono::duration<double>& duration) {
                    printf("%.3f ms", std::chrono::duration<double, std::milli>(duration).count());
                },
                [&](const my_units::bytes& bytes) {
                    if (!(bytes >> 10))
                        printf("%zu B", bytes);
                    else if (!(bytes >> 10))
                        printf("%zu KiB", bytes >> 10);
                    else
                        printf("%zu MiB", bytes >> 20);
                },
                [&](const CustomUnit<double>& customUnit) {
                    printf("%.3f %s", customUnit.value, customUnit.unit.c_str());
                },
                [&](const CustomUnit<int64_t>& customUnit) {
                    printf("%" PRId64 " %s", customUnit.value, customUnit.unit.c_str());
                } },
            element.value);
        printf("\n");
    }
}

uint32 StatsRecorder::get_frame_timestamp() const
{
    return cast<uint32>(frames.size());
}

double StatsRecorder::get_value_in_frame(uint32 frameTimestamp, const std::string& name) const
{
    if (frames.size() <= frameTimestamp)
        return 0;

    for (const auto& element : frames[frameTimestamp].elements) {
        if (element.name == name) {
            return std::visit(
                Utils::visitor {
                    [](const std::chrono::duration<double>& duration) -> double {
                        const std::chrono::duration<double, std::milli> durationInMs { duration };
                        return durationInMs.count();
                    },
                    [](const my_units::bytes& storageAmount) -> double {
                        // return (double)storageAmount.value();
                        return (double)storageAmount;
                    },
                    [](const CustomUnit<double>& customUnit) -> double {
                        return customUnit.value;
                    },
                    [](const CustomUnit<int64_t>& customUnit) -> double {
                        return (double)customUnit.value;
                    } },
                element.value);
        }
    }

    return 0;
}
