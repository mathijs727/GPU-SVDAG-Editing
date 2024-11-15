#include "timings.h"
#include <algorithm>

#if CAPTURE_GPU_TIMINGS

GPUTimingsManager::GPUTimingsManager()
{
    // Create all events ahead of time to prevent slowdowns.
    static constexpr size_t numEvents = 512;
    events.resize(numEvents);
    for (size_t i = 0; i < numEvents; ++i) {
        cudaEventCreate(&events[i]);
    }
}

GPUTimingsManager::~GPUTimingsManager()
{
    PROFILE_FUNCTION();
    for (auto ev : events) {
        cudaEventDestroy(ev);
    }
}

void GPUTimingsManager::startTiming(std::string name, cudaStream_t stream)
{
    if (currentEvent >= events.size() - 2) {
        printf("Running out of events, creating more...\n");
        const auto oldSize = events.size();
        events.resize(2 * oldSize);
        for (size_t i = 0; i < oldSize; ++i)
            cudaEventCreate(&events[oldSize + i]);
    }

    TimingSection section;
    section.start = events[currentEvent++];
    section.end = events[currentEvent++];
    cudaEventRecord(section.start, stream);
    lookup[name].push_back(section);
}

void GPUTimingsManager::endTiming(std::string name, cudaStream_t stream)
{
    const auto ev = lookup[name].back().end;
    cudaEventRecord(ev, stream);
}

void GPUTimingsManager::flush(StatsRecorder& statsRecorder) const
{
    PROFILE_FUNCTION();
    {
        PROFILE_SCOPE("cudaDeviceSynchronize()");
        cudaDeviceSynchronize();
    }
    for (const auto& [name, sections] : lookup) {
        std::chrono::duration<double> accumulatedDuration { 0 };
        for (const auto& timingSection : sections) {
            //cudaEventSynchronize(timingSection.end);
            float tmp;
            cudaEventElapsedTime(&tmp, timingSection.start, timingSection.end);
            std::chrono::duration<double, std::milli> duration { tmp };
            accumulatedDuration += duration;
        }
        statsRecorder.report(name, accumulatedDuration, Device::GPU);
    }
}

std::chrono::duration<double> GPUTimingsManager::getBlocking(std::string name) const
{
    check(lookup.find(name) != std::end(lookup));

    auto timingSection = lookup.find(name)->second.back();
    cudaEventSynchronize(timingSection.end);
    float tmp;
    cudaEventElapsedTime(&tmp, timingSection.start, timingSection.end);
    return std::chrono::duration<double, std::milli>(tmp);
}

void GPUTimingsManager::print() const
{
    PROFILE_FUNCTION();
    printf("GPU Timings:\n");
    std::vector<std::pair<float, std::string>> sortedSections;
    for (const auto& [name, sections] : lookup) {
        float timeInMs = 0.0f;
        for (const auto& timingSection : sections) {
            cudaEventSynchronize(timingSection.end);
            float tmp;
            cudaEventElapsedTime(&tmp, timingSection.start, timingSection.end);
            timeInMs += tmp;
        }
        sortedSections.push_back({ timeInMs, name });
    }

    std::sort(std::begin(sortedSections), std::end(sortedSections));
     
    //float totalTimeInMs = 0.0f;
    for (const auto& [timeInMs, name] : sortedSections) {
        printf("%-60s\t%f ms\n", name.c_str(), timeInMs);
        //totalTimeInMs += timeInMs;
    }
    //printf("=== TOTAL: %f ms ===\n", totalTimeInMs);
    //NOTE(Mathijs): incorrect when timing scopes overlap.
}

GPUTimingsManager::TimingScope GPUTimingsManager::timeScope(std::string name, cudaStream_t stream)
{
    startTiming(name, stream);

    TimingScope out {};
    out.pParent = this;
    out.name = name;
    out.stream = stream;
    return out;
}

GPUTimingsManager::TimingScope::~TimingScope()
{
    pParent->endTiming(name, stream);
}

#endif