#pragma once
#include "typedefs.h"
#include <chrono>
#include <cuda.h>
#include <deque>
#include <string_view>
#include <unordered_map>

class EventsManager;
class AsyncTimingSection {
public:
    AsyncTimingSection() = default;
    AsyncTimingSection(AsyncTimingSection&&);
    ~AsyncTimingSection();

private:
    friend class EventsManager;
    cudaEvent_t begin, end;
    EventsManager* pManager = nullptr;
    std::string_view name;
};

class EventsManager {
public:
    ~EventsManager();

    // Insert a fence on the GPU which sets the value to the given fenceValue when reached.
    // Use getLastCompletedFenceValue() to get the last fenceValue which the GPU processed.
    void insertFenceValue(const char*, size_t fenceValue);
    size_t getLastCompletedFenceValue(const char*);

    // RAII GPU timer on the main CUDA stream (automatically starts timer when createTiming is called
    //  and stops timer when the returned object goes out of scope).
    // You can get the elapsed time of the last completed event (of the given name) by calling getLastCompletedTiming.
    AsyncTimingSection createTiming(const char*);
    std::chrono::duration<double> getLastCompletedTiming(const char*);

private:
    friend class AsyncTimingSection;
    cudaEvent_t getEvent();
    void addSubmittedTiming(const AsyncTimingSection&);

private:
    // For each named event there may be multiple invocations in-flight.
    // This may happen when the CPU is multiple frames of the GPU.
    struct Timing {
        cudaEvent_t begin, end;
    };
    std::unordered_map<std::string_view, std::deque<Timing>> m_namedTimings;

    // Track for each fence the current value and the in-flight events that may update it.
    struct InFlightFence {
        cudaEvent_t ev;
        size_t newFenceValue;
    };
    struct Fence {
        size_t fenceValue;
        std::deque<InFlightFence> inFlightFences;
    };
    std::unordered_map<std::string_view, Fence> m_namedFences;

    std::deque<cudaEvent_t> m_recycledEvents;
    // Events that are used to signal the start/end of a frame.
    // This helps to ensure that we don't submit an infinite amount of work when submission (CPU) is faster than processing (GPU).
    std::deque<cudaEvent_t> m_frameEvents;
};
