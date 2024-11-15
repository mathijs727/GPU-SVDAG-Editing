#pragma once
#include "stats.h"
#include <chrono>
#include <cuda_runtime.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#if CAPTURE_GPU_TIMINGS

class GPUTimingsManager {
public:
    GPUTimingsManager();
    ~GPUTimingsManager();

    void startTiming(std::string name, cudaStream_t stream);
    void endTiming(std::string name, cudaStream_t stream);

    void flush(StatsRecorder& statsRecorder) const;
    std::chrono::duration<double> getBlocking(std::string name) const;
    void print() const;

    struct TimingScope {
    public:
        ~TimingScope();

    private:
        friend class GPUTimingsManager;
        GPUTimingsManager* pParent;
        std::string name;
        cudaStream_t stream;
    };
    TimingScope timeScope(std::string name, cudaStream_t stream);

private:
    struct TimingSection {
        cudaEvent_t start, end;
    };
    std::unordered_map<std::string, std::vector<TimingSection>> lookup;

    std::vector<cudaEvent_t> events;
    size_t currentEvent = 0;
};

#else // CAPTURE_GPU_TIMINGS

class GPUTimingsManager {
public:
    void startTiming(std::string, cudaStream_t) { }
    void endTiming(std::string, cudaStream_t) { }

    void flush(StatsRecorder& statsRecorder) const { }
    std::chrono::duration<double> getBlocking(std::string name) const { return { }; }
    void print() const { }

    struct TimingScope {
        int _dummy;
    };
    TimingScope timeScope(std::string, cudaStream_t) { return {}; }
};

#endif // CAPTURE_GPU_TIMINGS