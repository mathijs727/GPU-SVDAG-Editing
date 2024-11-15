#pragma once
#include "typedefs.h"
#include "my_units.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#define JSON_HAS_RANGES 0
#include <nlohmann/json.hpp>

enum class Device {
    CPU,
    GPU
};

nlohmann::json getSystemInfoJson();
nlohmann::json getDefineInfoJson();

class StatsRecorder {
public:
    StatsRecorder();

    void next_frame();
    template <typename T>
    void report(std::string name, T other) {
        checkAlways(false);
    }
    void report(std::string name, const std::chrono::duration<double>& duration, Device device);
    void report(std::string name, const my_units::bytes storageAmount, Device device);
    void reportFloat(std::string name, double value, std::string unit, Device device);
    void reportInt(std::string name, int64_t value, std::string unit, Device device);
    void clear();

    void write_csv();
    void write_csv(std::ostream& stream);

    void write_json();
    void write_json(std::ostream& stream);

    void printLastFrame() const;

    uint32 get_frame_timestamp() const;
    double get_value_in_frame(uint32 frameTimestamp, const std::string& name) const;

private:
    template <typename T>
    struct CustomUnit {
        std::string unit;
        T value;
    };
    struct Element {
        Device device;
        std::string name;
        std::variant<std::chrono::duration<double>, my_units::bytes, CustomUnit<double>, CustomUnit<int64_t>> value;
    };
    struct Frame {
        std::vector<Element> elements;
    };

    std::vector<Frame> frames;
};

enum class EStatNames {
    EarlyExitChecks,
    EntirelyFull_AddColors,
    SkipEdit_CopyColors,
    LeafEdit,
    FindOrAddLeaf,
    InteriorEdit,
    FindOrAddInterior,
    Max
};

inline std::string get_stat_name(EStatNames name)
{
    switch (name) {
#define C(a, b) \
    case a:     \
        return b;
        C(EStatNames::EarlyExitChecks, "early exit checks");
        C(EStatNames::EntirelyFull_AddColors, "entirely full - add colors");
        C(EStatNames::SkipEdit_CopyColors, "skip edit - copy colors");
        C(EStatNames::LeafEdit, "leaf edit");
        C(EStatNames::FindOrAddLeaf, "find or add leaf");
        C(EStatNames::InteriorEdit, "interior edit");
        C(EStatNames::FindOrAddInterior, "find or add interior");
#undef C
    case EStatNames::Max:
    default:
        check(false);
        return "";
    }
}

class LocalStatsRecorder {
public:
    explicit LocalStatsRecorder(StatsRecorder& statsRecorder)
        : statsRecorder(&statsRecorder)
    {
    }
    LocalStatsRecorder(LocalStatsRecorder& localStatsRecorder)
        : localStatsRecorder(&localStatsRecorder)
    {
    }

    ~LocalStatsRecorder()
    {
        check((statsRecorder && !localStatsRecorder) || (!statsRecorder && localStatsRecorder));
        if (statsRecorder) {
            for (uint32 statName = 0; statName < uint32(EStatNames::Max); statName++) {
                if (cumulativeElements[statName].num == 0)
                    continue;
                const auto name = get_stat_name(EStatNames(statName));
                statsRecorder->report(name, cumulativeElements[statName].time, cumulativeElements[statName].device);
                statsRecorder->report(name + " average", cumulativeElements[statName].time / cumulativeElements[statName].num, cumulativeElements[statName].device);
            }
        } else {
            for (uint32 statName = 0; statName < uint32(EStatNames::Max); statName++) {
                auto& element = localStatsRecorder->cumulativeElements[statName];
                element.time += cumulativeElements[statName].time;
                element.num += cumulativeElements[statName].num;
            }
        }
    }

    inline void report(EStatNames name, std::chrono::duration<double> time, Device device)
    {
        auto& element = cumulativeElements[uint32(name)];
        element.time += time;
        element.num++;
        element.device = device;
    }

private:
    StatsRecorder* statsRecorder = nullptr;
    LocalStatsRecorder* localStatsRecorder = nullptr;
    struct LocalStat {
        std::chrono::duration<double> time { 0 };
        int32 num = 0;
        Device device;
    };
    LocalStat cumulativeElements[uint32(EStatNames::Max)];
};

inline std::string make_prefix(size_t num)
{
    std::ostringstream s;
    for (size_t index = 0; index < num; index++) {
        s << "\t";
    }
    return s.str();
}

class BasicStats {
public:
    ~BasicStats()
    {
        check(name.empty());
    }

    inline void start_work(const std::string& workName)
    {
        check(name.empty());
        name = workName;
        time = std::chrono::high_resolution_clock::now();
    }
    template <typename T>
    inline double flush(T& recorder)
    {
        PROFILE_FUNCTION();
        const auto currentTime = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> timeInMs { currentTime - time };
        // ensure(timeInMs > 0.00001);
        if (ensure(!name.empty())) {
            recorder.report(name, timeInMs, Device::CPU);
        }
        name.clear();
        return timeInMs.count();
    }

private:
    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> time;
};

class Stats {
public:
    Stats()
        : prefix(get_prefix())
    {
        stack.push_back(this);
    }
    ~Stats()
    {
        flush();
        check(stack.back() == this);
        stack.pop_back();
    }

    FORCEINLINE void start_work(const char* workName)
    {
        flush();
        name = workName;
        time = std::chrono::high_resolution_clock::now();
    }
    FORCEINLINE void start_level_work(uint32 level, const std::string& workName)
    {
        std::ostringstream s;
        s << "level " << level << ": " << workName;
        start_work(s.str().c_str());
    }
    FORCEINLINE double flush(bool print = true)
    {
        PROFILE_FUNCTION();
        const auto currentTime = std::chrono::high_resolution_clock::now();
        const double timeInMs = double((currentTime - time).count()) / 1.e6;
        if (!name.empty() && print) {
            std::cout << prefix << name << " took " << timeInMs << "ms" << std::endl;
        }
        name.clear();
        return timeInMs;
    }

    inline static std::string get_prefix()
    {
        return make_prefix(stack.size());
    }

private:
    const std::string prefix;
    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> time;

    static std::vector<Stats*> stack;
};

#define PASTE_HELPER(a, b) a##b
#define PASTE(a, b) PASTE_HELPER(a, b)
#define SCOPED_STATS(name)        \
    printf(name "...\n");         \
    Stats PASTE(stats, __LINE__); \
    PASTE(stats, __LINE__).start_work(name);

struct SimpleScopeStat {
    SimpleScopeStat()
        : startTime(std::chrono::high_resolution_clock::now())
    {
    }

    inline std::chrono::duration<double> get_time() const
    {
        return std::chrono::high_resolution_clock::now() - startTime;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

struct EditScopeStat {
#if EDITS_PROFILING
    EditScopeStat(LocalStatsRecorder& recorder, EStatNames name)
        : recorder(recorder)
        , name(name)
    {
        startTime = std::chrono::high_resolution_clock::now();
    }
    ~EditScopeStat()
    {
        if (!paused) {
            const auto time = get_time();
            recorder.report(name, time);
        }
    }

    inline void pause()
    {
        const auto time = get_time();

        check(!paused);
        paused = true;
        recorder.report(name, time);
    }

    inline void resume()
    {
        check(paused);
        paused = false;
        startTime = std::chrono::high_resolution_clock::now();
    }

private:
    LocalStatsRecorder& recorder;
    const EStatNames name;
    bool paused = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    inline double get_time() const
    {
        const auto currentTime = std::chrono::high_resolution_clock::now();
        const double timeInMs = double((currentTime - startTime).count()) / 1.e6;
        return timeInMs;
    }
#else
    EditScopeStat(LocalStatsRecorder& recorder, EStatNames name)
    {
    }
    inline void pause()
    {
    }

    inline void resume()
    {
    }
#endif
};


std::string getHostNameCpp();
std::string getOperatingSystemName();
