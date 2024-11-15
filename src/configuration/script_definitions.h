#pragma once
#include "hash_dag_enum.h"

#define SCENE "epiccitadel" // The name of the scene to load at start-up (see data folder)
#define SCENE_DEPTH 17 // The number of SVDAG levels of the scene.
#define SAVE_SCENE 0 // Loading a scene requires an expensive color => material conversion step. Store the result (in the data folder) to accelerate subsequent loads.

#define DAG_TYPE EDag::MyGpuDag // The type of DAG: static, the HashDAG, or our editable GPU DAG.
#define EDITS_ENABLE_COLORS 1 // Whether to the colors associated with the SVDAG file (applicable to all DAG types).
#define EDITS_ENABLE_MATERIALS 1 // Whether to enable materials (our GPU DAG only).

#define USE_REPLAY 0 // Whether to load and play a replay file.
#define REPLAY_NAME "large_edits" // Name of the replay (see replays folder).
#define REPLAY_DEPTH 16 // The number of SVDAG levels when the replay was recorded.

#define AUTO_GARBAGE_COLLECT 0 // Whether to automatically run garbage collection after each edit (our GPU DAG only).
#define UNDO_REDO 1 // Enable undo/redo functionality.
#define ENABLE_CHECKS 0 // Enable debug assertions.
#define CAPTURE_GPU_TIMINGS 1 // Capture GPU timings and display them in the User Interface
#define CAPTURE_MEMORY_STATS_SLOW 0 // Capture additional statistics about memory usage; may slow down performance.