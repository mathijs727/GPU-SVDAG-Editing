#include "replay.h"
#include "dags/hash_dag/hash_dag_editors.h"
#include "dags/my_gpu_dags/my_gpu_dag_editors.h"
#include "engine.h"

struct MyLossyColors;
template <bool>
struct MyLosslessColors;

constexpr float scalingFactor = float(1 << SCENE_DEPTH) / float(1 << REPLAY_DEPTH);

void ReplayActionUndo::apply()
{
#if UNDO_REDO
    if (pEngine->hashDag.is_valid())
        pEngine->undoRedo.undo(pEngine->hashDag, pEngine->hashDagColors);
    if (pEngine->myGpuHashDag.is_valid())
        pEngine->gpuUndoRedo.undo(pEngine->myGpuHashDag);
#endif
}

void ReplayActionRedo::apply()
{
#if UNDO_REDO
    if (pEngine->hashDag.is_valid())
        pEngine->undoRedo.redo(pEngine->hashDag, pEngine->hashDagColors);
    if (pEngine->myGpuHashDag.is_valid())
        pEngine->gpuUndoRedo.redo(pEngine->myGpuHashDag);
#endif
}

void ReplayActionSetLocation::apply()
{
    pEngine->view.position = location;
}

void ReplayActionSetRotation::apply()
{
    pEngine->view.rotation = rotation;
}

void ReplayActionSetToolParameters::apply()
{
    pEngine->config.path.centerPath = truncate(make_float3(toolPath.centerPath) * scalingFactor);
    pEngine->config.path.neighbourPath = make_uint3(make_int3(pEngine->config.path.centerPath) + (make_int3(toolPath.neighbourPath) - make_int3(toolPath.centerPath)));
    pEngine->config.copySourcePath = truncate(make_float3(copySourcePath) * scalingFactor);
    pEngine->config.copyDestPath = truncate(make_float3(copyDestPath) * scalingFactor);
    pEngine->config.radius = radius * scalingFactor;
    pEngine->config.tool = ETool(tool);
}

void ReplayActionSphere::apply()
{
    if (add) {
        pEngine->edit<SphereEditor<true>>(
            location * scalingFactor,
            radius * scalingFactor);
        pEngine->edit<MyGpuSphereEditor<true>>(
            location * scalingFactor,
            radius * scalingFactor,
            editMaterial);
    } else {
        pEngine->edit<SphereEditor<false>>(
            location * scalingFactor,
            radius * scalingFactor);
        pEngine->edit<MyGpuSphereEditor<false>>(
            location * scalingFactor,
            radius * scalingFactor,
            editMaterial);
    }
}

void ReplayActionCube::apply()
{
    if (add) {
        pEngine->edit<BoxEditor<true>>(
            location * scalingFactor,
            radius * scalingFactor);
        pEngine->edit<MyGpuBoxEditor<true>>(
            location * scalingFactor,
            radius * scalingFactor,
            editMaterial);
    } else {
        pEngine->edit<BoxEditor<false>>(
            location * scalingFactor,
            radius * scalingFactor);
        pEngine->edit<MyGpuBoxEditor<false>>(
            location * scalingFactor,
            radius * scalingFactor,
            editMaterial);
    }
}

void ReplayActionCopy::apply()
{
    pEngine->edit<CopyEditor>(
        pEngine->hashDag,
        pEngine->hashDagColors,
        src * scalingFactor,
        dest * scalingFactor,
        location * scalingFactor,
        radius * scalingFactor,
        transform,
        pEngine->statsRecorder,
        enableSwirl,
        swirlPeriod);
    pEngine->edit<MyGpuCopyEditor<MyGPUHashDAG<EMemoryType::GPU_Malloc>>>(
        pEngine->myGpuHashDag,
        src * scalingFactor,
        dest * scalingFactor,
        location * scalingFactor,
        radius * scalingFactor);
}

void ReplayActionFill::apply()
{
    pEngine->edit<FillEditorColors>(
        pEngine->hashDag,
        pEngine->hashDagColors,
        location * scalingFactor,
        radius * scalingFactor);
}

void ReplayActionPaint::apply()
{
    pEngine->edit<SpherePaintEditor>(
        location * scalingFactor,
        radius * scalingFactor);
    pEngine->edit<MyGpuSpherePaintEditor>(
        location * scalingFactor,
        radius * scalingFactor,
        paintColor,
        paintMaterial);
}
