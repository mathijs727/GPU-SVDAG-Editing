#pragma once
#include "memory.h"

template <EMemoryType>
struct MyGPUHashDAG;
struct MyLossyColors;
template <bool>
struct MyLosslessColors;
class StatsRecorder;
class MyGPUHashDAGUndoRedo;

template <typename Editor>
void editMyHashDag(const Editor& editor, MyGPUHashDAG<EMemoryType::GPU_Malloc>& dag, MyGPUHashDAGUndoRedo& undoRedo, StatsRecorder& statsRecorder, GpuMemoryPool& memPool, cudaStream_t stream);
