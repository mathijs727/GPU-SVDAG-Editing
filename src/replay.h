#pragma once
#include "gmath/Matrix3x3.h"
#include "gmath/Vector3.h"
#include "tracer.h"
#include "typedefs.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

class Engine;

enum class EReplayActionType {
    SetLocation,
    SetRotation,
    SetToolParameters,
    EditSphere,
    EditCube,
    EditCopy,
    EditFill,
    EditPaint,
    Undo,
    Redo,
    EndFrame,
};

inline std::string replay_action_to_string(EReplayActionType action)
{
    switch (action) {
    case EReplayActionType::SetLocation:
        return "SetLocation";
    case EReplayActionType::SetRotation:
        return "SetRotation";
    case EReplayActionType::SetToolParameters:
        return "SetToolParameters";
    case EReplayActionType::EditSphere:
        return "EditSphere";
    case EReplayActionType::EditCube:
        return "EditCube";
    case EReplayActionType::EditCopy:
        return "EditCopy";
    case EReplayActionType::EditFill:
        return "EditFill";
    case EReplayActionType::EditPaint:
        return "EditPaint";
    case EReplayActionType::Undo:
        return "Undo";
    case EReplayActionType::Redo:
        return "Redo";
    case EReplayActionType::EndFrame:
        return "EndFrame";
    default:
        checkAlways(false);
        return "";
    }
}

inline EReplayActionType string_to_replay_action(const std::string& string)
{
    if (string == "SetLocation")
        return EReplayActionType::SetLocation;
    if (string == "SetRotation")
        return EReplayActionType::SetRotation;
    if (string == "SetToolParameters")
        return EReplayActionType::SetToolParameters;
    if (string == "EditSphere")
        return EReplayActionType::EditSphere;
    if (string == "EditCube")
        return EReplayActionType::EditCube;
    if (string == "EditCopy")
        return EReplayActionType::EditCopy;
    if (string == "EditFill")
        return EReplayActionType::EditFill;
    if (string == "EditPaint")
        return EReplayActionType::EditPaint;
    if (string == "Undo")
        return EReplayActionType::Undo;
    if (string == "Redo")
        return EReplayActionType::Redo;
    if (string == "EndFrame")
        return EReplayActionType::EndFrame;
    checkAlways(false);
    return EReplayActionType::SetLocation;
}

class IReplayAction {
public:
    const EReplayActionType type;
    Engine* pEngine;

    IReplayAction(EReplayActionType type, Engine* pEngine)
        : type(type)
        , pEngine(pEngine)
    {
    }
    virtual ~IReplayAction() = default;

    virtual void load(const std::vector<std::string>& data) = 0;
    virtual void write(std::vector<std::string>& data) = 0;
    virtual void apply() = 0;
};

template <EReplayActionType Type>
class TReplayAction : public IReplayAction {
public:
    TReplayAction(Engine* pEngine)
        : IReplayAction(Type, pEngine)
    {
    }
};

class ReplayActionEndFrame : public TReplayAction<EReplayActionType::EndFrame> {
public:
    ReplayActionEndFrame(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }

    virtual void load(const std::vector<std::string>& data) override { }
    virtual void write(std::vector<std::string>& data) override { }
    virtual void apply() override { }
};

class ReplayActionUndo : public TReplayAction<EReplayActionType::Undo> {
public:
    ReplayActionUndo(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }

    virtual void load(const std::vector<std::string>& data) override { }
    virtual void write(std::vector<std::string>& data) override { }
    virtual void apply() override;
};

class ReplayActionRedo : public TReplayAction<EReplayActionType::Redo> {
public:
    ReplayActionRedo(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }

    virtual void load(const std::vector<std::string>& data) override { }
    virtual void write(std::vector<std::string>& data) override { }
    virtual void apply() override;
};

class ReplayActionSetLocation : public TReplayAction<EReplayActionType::SetLocation> {
public:
    ReplayActionSetLocation(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }
    ReplayActionSetLocation(Vector3 location, Engine* pEngine)
        : TReplayAction(pEngine)
        , location(location)
    {
    }

    Vector3 location;

    virtual void load(const std::vector<std::string>& data) override
    {
        checkAlways(data.size() == 3);
        location.X = std::stof(data[0]);
        location.Y = std::stof(data[1]);
        location.Z = std::stof(data[2]);
    }
    virtual void write(std::vector<std::string>& data) override
    {
        data.push_back(std::to_string(location.X));
        data.push_back(std::to_string(location.Y));
        data.push_back(std::to_string(location.Z));
    }

    virtual void apply() override;
};

class ReplayActionSetRotation : public TReplayAction<EReplayActionType::SetRotation> {
public:
    ReplayActionSetRotation(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }
    ReplayActionSetRotation(Matrix3x3 rotation, Engine* pEngine)
        : TReplayAction(pEngine)
        , rotation(rotation)
    {
    }

    Matrix3x3 rotation;

    virtual void load(const std::vector<std::string>& data) override
    {
        checkAlways(data.size() == 9);
        rotation.D00 = std::stof(data[0]);
        rotation.D01 = std::stof(data[1]);
        rotation.D02 = std::stof(data[2]);
        rotation.D10 = std::stof(data[3]);
        rotation.D11 = std::stof(data[4]);
        rotation.D12 = std::stof(data[5]);
        rotation.D20 = std::stof(data[6]);
        rotation.D21 = std::stof(data[7]);
        rotation.D22 = std::stof(data[8]);
    }
    virtual void write(std::vector<std::string>& data) override
    {
        data.push_back(std::to_string(rotation.D00));
        data.push_back(std::to_string(rotation.D01));
        data.push_back(std::to_string(rotation.D02));
        data.push_back(std::to_string(rotation.D10));
        data.push_back(std::to_string(rotation.D11));
        data.push_back(std::to_string(rotation.D12));
        data.push_back(std::to_string(rotation.D20));
        data.push_back(std::to_string(rotation.D21));
        data.push_back(std::to_string(rotation.D22));
    }

    virtual void apply() override;
};

class ReplayActionSetToolParameters : public TReplayAction<EReplayActionType::SetToolParameters> {
public:
    ReplayActionSetToolParameters(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }
    ReplayActionSetToolParameters(const ToolPath& toolPath, uint3 copySourcePath, uint3 copyDestPath, float radius, uint32 tool, Engine* pEngine)
        : TReplayAction(pEngine)
        , toolPath(toolPath)
        , copySourcePath(copySourcePath)
        , copyDestPath(copyDestPath)
        , radius(radius)
        , tool(tool)
    {
    }

    ToolPath toolPath {};
    uint3 copySourcePath;
    uint3 copyDestPath;
    float radius {};
    uint32 tool {};

    virtual void load(const std::vector<std::string>& data) override
    {
        checkAlways(data.size() == 11 || data.size() == 14);
        toolPath.centerPath.x = (uint32)std::stoi(data[0]);
        toolPath.centerPath.y = (uint32)std::stoi(data[1]);
        toolPath.centerPath.z = (uint32)std::stoi(data[2]);
        if (data.size() > 11) {
            toolPath.neighbourPath.x = (uint32)std::stoi(data[3]);
            toolPath.neighbourPath.y = (uint32)std::stoi(data[4]);
            toolPath.neighbourPath.z = (uint32)std::stoi(data[5]);
        }
        const int i = data.size() == 11 ? 3 : 6;
        copySourcePath.x = (uint32)std::stoi(data[i + 0]);
        copySourcePath.y = (uint32)std::stoi(data[i + 1]);
        copySourcePath.z = (uint32)std::stoi(data[i + 2]);
        copyDestPath.x = (uint32)std::stoi(data[i + 3]);
        copyDestPath.y = (uint32)std::stoi(data[i + 4]);
        copyDestPath.z = (uint32)std::stoi(data[i + 5]);
        radius = std::stof(data[i + 6]);
        tool = (uint32)std::stoi(data[i + 7]);
    }
    virtual void write(std::vector<std::string>& data) override
    {
        data.push_back(std::to_string(toolPath.centerPath.x));
        data.push_back(std::to_string(toolPath.centerPath.y));
        data.push_back(std::to_string(toolPath.centerPath.z));
        data.push_back(std::to_string(toolPath.neighbourPath.x));
        data.push_back(std::to_string(toolPath.neighbourPath.y));
        data.push_back(std::to_string(toolPath.neighbourPath.z));
        data.push_back(std::to_string(copySourcePath.x));
        data.push_back(std::to_string(copySourcePath.y));
        data.push_back(std::to_string(copySourcePath.z));
        data.push_back(std::to_string(copyDestPath.x));
        data.push_back(std::to_string(copyDestPath.y));
        data.push_back(std::to_string(copyDestPath.z));
        data.push_back(std::to_string(radius));
        data.push_back(std::to_string(tool));
    }

    virtual void apply() override;
};

class ReplayActionSphere : public TReplayAction<EReplayActionType::EditSphere> {
public:
    ReplayActionSphere(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }
    ReplayActionSphere(float3 location, float radius, bool add, uint32_t editMaterial, Engine* pEngine)
        : TReplayAction(pEngine)
        , location(location)
        , radius(radius)
        , add(add)
        , editMaterial(editMaterial)
    {
    }

    float3 location {};
    float radius = 0;
    bool add = false;
    uint32_t editMaterial = 1;

    virtual void load(const std::vector<std::string>& data) override
    {
        checkAlways(data.size() == 5 || data.size() == 6);
        location.x = std::stof(data[0]);
        location.y = std::stof(data[1]);
        location.z = std::stof(data[2]);
        radius = std::stof(data[3]);
        checkAlways(data[4] == "true" || data[4] == "false" || data[4] == "TRUE" || data[4] == "FALSE");
        add = data[4] == "true" || data[4] == "TRUE";
        if (data.size() > 5)
            editMaterial = std::stoi(data[5]);
    }
    virtual void write(std::vector<std::string>& data) override
    {
        data.push_back(std::to_string(location.x));
        data.push_back(std::to_string(location.y));
        data.push_back(std::to_string(location.z));
        data.push_back(std::to_string(radius));
        data.push_back(add ? "true" : "false");
        data.push_back(std::to_string(editMaterial));
    }

    virtual void apply() override;
};

class ReplayActionCube : public TReplayAction<EReplayActionType::EditCube> {
public:
    ReplayActionCube(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }
    ReplayActionCube(float3 location, float radius, bool add, uint32_t editMaterial, Engine* pEngine)
        : TReplayAction(pEngine)
        , location(location)
        , radius(radius)
        , add(add)
        , editMaterial(editMaterial)
    {
    }

    float3 location {};
    float radius = 0;
    bool add = false;
    uint32_t editMaterial = 1;

    virtual void load(const std::vector<std::string>& data) override
    {
        checkAlways(data.size() == 5 || data.size() == 6);
        location.x = std::stof(data[0]);
        location.y = std::stof(data[1]);
        location.z = std::stof(data[2]);
        radius = std::stof(data[3]);
        checkAlways(data[4] == "true" || data[4] == "false" || data[4] == "TRUE" || data[4] == "FALSE");
        add = data[4] == "true" || data[4] == "TRUE";
        if (data.size() > 5)
            editMaterial = std::stoi(data[5]);
    }
    virtual void write(std::vector<std::string>& data) override
    {
        data.push_back(std::to_string(location.x));
        data.push_back(std::to_string(location.y));
        data.push_back(std::to_string(location.z));
        data.push_back(std::to_string(radius));
        data.push_back(add ? "true" : "false");
        data.push_back(std::to_string(editMaterial));
    }

    virtual void apply() override;
};

class ReplayActionCopy : public TReplayAction<EReplayActionType::EditCopy> {
public:
    ReplayActionCopy(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }
    ReplayActionCopy(float3 location, float3 src, float3 dest, float radius, const Matrix3x3& transform, bool enableSwirl, float swirlPeriod, Engine* pEngine)
        : TReplayAction(pEngine)
        , location(location)
        , src(src)
        , dest(dest)
        , radius(radius)
        , transform(transform)
        , enableSwirl(enableSwirl)
        , swirlPeriod(swirlPeriod)
    {
    }

    float3 location {};
    float3 src {};
    float3 dest {};
    float radius = 0;
    Matrix3x3 transform;
    bool enableSwirl = false;
    float swirlPeriod = 1;

    virtual void load(const std::vector<std::string>& data) override
    {
        checkAlways(data.size() == 10 || data.size() == 21);
        location.x = std::stof(data[0]);
        location.y = std::stof(data[1]);
        location.z = std::stof(data[2]);
        src.x = std::stof(data[3]);
        src.y = std::stof(data[4]);
        src.z = std::stof(data[5]);
        dest.x = std::stof(data[6]);
        dest.y = std::stof(data[7]);
        dest.z = std::stof(data[8]);
        radius = std::stof(data[9]);
        if (data.size() == 21) {
            transform.D00 = std::stof(data[10]);
            transform.D01 = std::stof(data[11]);
            transform.D02 = std::stof(data[12]);
            transform.D10 = std::stof(data[13]);
            transform.D11 = std::stof(data[14]);
            transform.D12 = std::stof(data[15]);
            transform.D20 = std::stof(data[16]);
            transform.D21 = std::stof(data[17]);
            transform.D22 = std::stof(data[18]);
            checkAlways(data[19] == "true" || data[19] == "false" || data[19] == "TRUE" || data[19] == "FALSE");
            enableSwirl = data[19] == "true" || data[19] == "TRUE";
            swirlPeriod = std::stof(data[20]);
        }
    }
    virtual void write(std::vector<std::string>& data) override
    {
        data.push_back(std::to_string(location.x));
        data.push_back(std::to_string(location.y));
        data.push_back(std::to_string(location.z));
        data.push_back(std::to_string(src.x));
        data.push_back(std::to_string(src.y));
        data.push_back(std::to_string(src.z));
        data.push_back(std::to_string(dest.x));
        data.push_back(std::to_string(dest.y));
        data.push_back(std::to_string(dest.z));
        data.push_back(std::to_string(radius));
        data.push_back(std::to_string(transform.D00));
        data.push_back(std::to_string(transform.D01));
        data.push_back(std::to_string(transform.D02));
        data.push_back(std::to_string(transform.D10));
        data.push_back(std::to_string(transform.D11));
        data.push_back(std::to_string(transform.D12));
        data.push_back(std::to_string(transform.D20));
        data.push_back(std::to_string(transform.D21));
        data.push_back(std::to_string(transform.D22));
        data.push_back(enableSwirl ? "true" : "false");
        data.push_back(std::to_string(swirlPeriod));
    }

    virtual void apply() override;
};

class ReplayActionFill : public TReplayAction<EReplayActionType::EditFill> {
public:
    ReplayActionFill(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }
    ReplayActionFill(float3 location, float radius, Engine* pEngine)
        : TReplayAction(pEngine)
        , location(location)
        , radius(radius)
    {
    }

    float3 location {};
    float radius = 0;

    virtual void load(const std::vector<std::string>& data) override
    {
        checkAlways(data.size() == 4);
        location.x = std::stof(data[0]);
        location.y = std::stof(data[1]);
        location.z = std::stof(data[2]);
        radius = std::stof(data[3]);
    }
    virtual void write(std::vector<std::string>& data) override
    {
        data.push_back(std::to_string(location.x));
        data.push_back(std::to_string(location.y));
        data.push_back(std::to_string(location.z));
        data.push_back(std::to_string(radius));
    }

    virtual void apply() override;
};

class ReplayActionPaint : public TReplayAction<EReplayActionType::EditPaint> {
public:
    ReplayActionPaint(Engine* pEngine)
        : TReplayAction(pEngine)
    {
    }
    ReplayActionPaint(float3 location, float radius, float3 paintColor, uint32_t paintMaterial, Engine* pEngine)
        : TReplayAction(pEngine)
        , location(location)
        , radius(radius)
        , paintColor(paintColor)
        , paintMaterial(paintMaterial)
    {
    }

    float3 location {};
    float radius = 0;
    float3 paintColor = make_float3(1, 1, 1);
    uint32_t paintMaterial = 1;

    virtual void load(const std::vector<std::string>& data) override
    {
        checkAlways(data.size() == 4 || data.size() == 7 || data.size() == 8);
        location.x = std::stof(data[0]);
        location.y = std::stof(data[1]);
        location.z = std::stof(data[2]);
        radius = std::stof(data[3]);
        if (data.size() > 6) {
            paintColor = make_float3(std::stof(data[4]), std::stof(data[5]), std::stof(data[6]));
        }
        if (data.size() > 7) {
            paintMaterial = std::stoi(data[7]);
        }
    }
    virtual void write(std::vector<std::string>& data) override
    {
        data.push_back(std::to_string(location.x));
        data.push_back(std::to_string(location.y));
        data.push_back(std::to_string(location.z));
        data.push_back(std::to_string(radius));
        data.push_back(std::to_string(paintColor.x));
        data.push_back(std::to_string(paintColor.y));
        data.push_back(std::to_string(paintColor.z));
        data.push_back(std::to_string(paintMaterial));
    }

    virtual void apply() override;
};

inline std::unique_ptr<IReplayAction> replay_action_factory(EReplayActionType type, Engine* pEngine)
{
    switch (type) {
    case EReplayActionType::SetLocation:
        return std::make_unique<ReplayActionSetLocation>(pEngine);
    case EReplayActionType::SetRotation:
        return std::make_unique<ReplayActionSetRotation>(pEngine);
    case EReplayActionType::SetToolParameters:
        return std::make_unique<ReplayActionSetToolParameters>(pEngine);
    case EReplayActionType::EditSphere:
        return std::make_unique<ReplayActionSphere>(pEngine);
    case EReplayActionType::EditCube:
        return std::make_unique<ReplayActionCube>(pEngine);
    case EReplayActionType::EditCopy:
        return std::make_unique<ReplayActionCopy>(pEngine);
    case EReplayActionType::EditFill:
        return std::make_unique<ReplayActionFill>(pEngine);
    case EReplayActionType::EditPaint:
        return std::make_unique<ReplayActionPaint>(pEngine);
    case EReplayActionType::Undo:
        return std::make_unique<ReplayActionUndo>(pEngine);
    case EReplayActionType::Redo:
        return std::make_unique<ReplayActionRedo>(pEngine);
    case EReplayActionType::EndFrame:
        return std::make_unique<ReplayActionEndFrame>(pEngine);
    default:
        checkAlways(false);
        return {};
    }
}

class ReplayManager {
public:
    inline ReplayManager() = default;
    inline ReplayManager(Engine* pEngine)
        : pEngine(pEngine)
    {
    }

    inline void replay_frame()
    {
        if (replayIndex < actions.size()) {
            frameIndex++;
            printf("Replaying frame %" PRIu64 "/%" PRIu64 "\n", uint64(frameIndex), uint64(numFrames));
            IReplayAction* action;
            do {
                action = actions[replayIndex].get();
                check(action);
                action->apply();
                replayIndex++;
            } while (action->type != EReplayActionType::EndFrame && ensure(replayIndex < actions.size()));
        }
    }

    template <typename T, typename... TArgs>
    inline void add_action(TArgs&&... args)
    {
        actions.push_back(std::make_unique<T>(std::forward<TArgs>(args)..., pEngine));
    }

    inline void write_csv()
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::stringstream fileName;
        fileName << SCENE << "_" << std::put_time(&tm, "%d-%m-%Y_%H-%M-%S.csv");

        std::filesystem::path path = ROOT_FOLDER;
        path = path / "replays" / fileName.str();

        std::ofstream os(path);
        checkAlways(os.is_open());

        write_csv(os);

        checkAlways(os.good());
        os.close();
    }
    inline void write_csv(std::ostream& stream)
    {
        for (auto& action : actions) {
            stream << replay_action_to_string(action->type);
            std::vector<std::string> data;
            action->write(data);
            for (auto& str : data) {
                stream << "," << str;
            }
            stream << "\n";
        }
    }
    inline void load_csv(const std::filesystem::path& path)
    {
        printf("Loading replay %s\n", path.string().c_str());

        std::ifstream is(path);
        checkfAlways(is.is_open() && is.good(), "Path: %s", path.string().c_str());

        frameIndex = 0;
        numFrames = 0;
        replayIndex = 0;
        actions.resize(0);

        load_csv(is);

        is.close();
    }
    inline void load_csv(std::ifstream& stream)
    {
        std::vector<std::string> data;

        char c;
        while (!(stream.get(c), stream.eof())) {
            if (c == '\n') {
                const EReplayActionType type = string_to_replay_action(data[0]);
                data.erase(data.begin());

                if (type == EReplayActionType::EndFrame)
                    numFrames++;

                // remove empty cells
                while (!data.empty() && data.back().empty())
                    data.pop_back();

                auto action = replay_action_factory(type, pEngine);
                checkAlways(action->type == type);
                action->load(data);
                actions.push_back(std::move(action));
                data.resize(0);
            } else if (c == ',') {
                data.emplace_back();
            } else {
                if (data.empty()) {
                    data.emplace_back();
                }
                data.back().push_back(c);
            }
        }

        check(data.empty());
    }

    inline void clear()
    {
        actions.resize(0);
    }
    inline void reset_replay()
    {
        frameIndex = 0;
        replayIndex = 0;
    }
    inline bool is_empty() const
    {
        return actions.empty();
    }
    inline bool at_end() const
    {
        return replayIndex == actions.size();
    }
    inline std::size_t num_frames() const
    {
        return numFrames;
    }
    inline Vector3 get_initial_position() const
    {
        for (auto& action : actions) {
            if (action->type == EReplayActionType::SetLocation) {
                return static_cast<const ReplayActionSetLocation&>(*action).location;
            }
        }
        return {};
    }
    inline Matrix3x3 get_initial_rotation() const
    {
        for (auto& action : actions) {
            if (action->type == EReplayActionType::SetRotation) {
                return static_cast<const ReplayActionSetRotation&>(*action).rotation;
            }
        }
        return {};
    }

private:
    Engine* pEngine;

    std::size_t frameIndex = 0;
    std::size_t numFrames = 0;
    std::size_t replayIndex = 0;
    std::vector<std::unique_ptr<IReplayAction>> actions;
};
