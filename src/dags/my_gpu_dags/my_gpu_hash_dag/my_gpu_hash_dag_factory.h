#pragma once
#include "my_gpu_hash_dag.h"

struct BasicDAG;
struct BasicDAGCompressedColors;
struct BasicDAGUncompressedColors;
struct VoxelTextures;

struct MyGPUHashDAGFactory {
    static void load_from_DAG(
        MyGPUHashDAG<EMemoryType::GPU_Malloc>& outDag, const BasicDAG& inDag, const BasicDAGCompressedColors& inColors, const VoxelTextures& inVoxelTextures);
    static void load_from_DAG(
        MyGPUHashDAG<EMemoryType::GPU_Malloc>& outDag, const BasicDAG& inDag, const BasicDAGUncompressedColors& inColors, const VoxelTextures& inVoxelTextures);
};