#pragma once
#include "hash_dag_enum.h"

#define TARGET_LOAD_FACTOR 96 // Determines the number of buckets in the hash table. Targets 96 items per bucket for the initial SVDAG.
#define HASH_TABLE_WARP_ADD 1 // Use warps to insert items into the hash tables (rather than individual threads)
#define HASH_TABLE_WARP_FIND 1 // Use warps to search items in the hash tables (rather than individual threads)
#define HASH_TABLE_ACCURATE_RESERVE_MEMORY 1 // Editing will grow the SVDAG; requiring more memory. Grow the allocators by counting how many nodes/leaves are created of each particular size.
#define HASH_TABLE_TYPE HashTableType::CompactAccelerationHash // The type of hash table used.
#define HASH_TABLE_HASH_METHOD HashMethod::SlabHashXor // The hash function to use; see `src/dags/my_gpu_dags/gpu_hash_table_base.h`.
#define HASH_TABLE_STORE_SLABS_IN_TABLE 1 // Store a contiguous array containing the first slab of each bucket, rather than pointers to the first slab.
#define HASH_DAG_MATERIAL_BITS 4 // The number of bits used to encode a material.