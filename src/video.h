#pragma once

#include "typedefs.h"
#include <filesystem>

class Engine;

class VideoManager
{
public:
    void load_video(const std::filesystem::path& path);
    void tick(Engine& engine);

private:
    struct Replay
    {
        std::filesystem::path replayPath;
        double timeAfterReplay = 0;
    };
    std::vector<Replay> replays;
    uint32 replayIndex = 0;
    double nextReplayTime = 0;
    bool isReplaying = false;
};