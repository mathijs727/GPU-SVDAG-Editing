import os
import shared
from shared import escape_string, load_build_tools

def benchmark_methods(scenes, replays, material_bits, profiling_folder):
    for scene, scene_depth in scenes:
        #for hash_table in ["Atomic64", "TicketBoard", "AccelerationHash", "CompactAccelerationHash"]:
        for hash_table in ["Atomic64", "TicketBoard", "AccelerationHash", "CompactAccelerationHash", "HashDag"]:
            defines = shared.default_script_definitions()
            defines["script_definitions.h"]["SCENE"] = escape_string(scene)
            defines["script_definitions.h"]["SCENE_DEPTH"] = scene_depth
            defines["script_definitions.h"]["EDITS_ENABLE_COLORS"] = 1 if material_bits > 0 else 0
            defines["script_definitions.h"]["EDITS_ENABLE_MATERIALS"] = 1 if material_bits > 0 else 0
            defines["script_definitions.h"]["CAPTURE_GPU_TIMINGS"] = 1
            defines["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 0
            if hash_table ==  "HashDag":
                defines["script_definitions.h"]["DAG_TYPE"] = f"EDag::HashDag"
                defines["gpu_hash_dag_definitions.h"]["HASH_TABLE_TYPE"] = f"HashTableType::Atomic64"
            else:
                defines["script_definitions.h"]["DAG_TYPE"] = f"EDag::MyGpuDag"
                defines["gpu_hash_dag_definitions.h"]["HASH_TABLE_TYPE"] = f"HashTableType::{hash_table}"
            defines["gpu_hash_dag_definitions.h"]["TARGET_LOAD_FACTOR"] = 96
            defines["gpu_hash_dag_definitions.h"]["HASH_TABLE_ENABLE_GARBAGE_COLLECTION"] = 0
            defines["gpu_hash_dag_definitions.h"]["HASH_DAG_MATERIAL_BITS"] = material_bits
            defines["profile_definitions.h"]["PROFILING_PATH"] = escape_string(profiling_folder.replace("\\", "/"))
            construction_profile_name = f"{scene}{2**(scene_depth-10)}k-{hash_table}-construction"
            defines["profile_definitions.h"]["CONSTRUCT_STATS_FILE_PATH"] = escape_string(construction_profile_name)

            # Initial construction, saving the hash dag.
            #if hash_table != "HashDag":
            #    defines["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 1
            #    defines["script_definitions.h"]["SAVE_SCENE"] = "1"
            #    shared.compile_and_run_hash_dag(defines)

            # Performance runs
            defines["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 0
            defines["script_definitions.h"]["SAVE_SCENE"] = "0"
            for replay, replay_depth in replays[scene]:
                defines["script_definitions.h"]["REPLAY_NAME"] = escape_string(replay)
                defines["script_definitions.h"]["REPLAY_DEPTH"] = str(replay_depth)
                for run in range(10):
                    profile_name = f"{scene}{2**(scene_depth-10)}k-{replay}-{hash_table}_run{run}"
                    defines["profile_definitions.h"]["STATS_FILES_PREFIX"] = escape_string(profile_name)
                    shared.compile_and_run_hash_dag(defines)
                    
            """# Memory runs
            auto_garbage_collect = hash_table != "HashDag"
            defines["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 1
            defines["script_definitions.h"]["AUTO_GARBAGE_COLLECT"] = 1 if auto_garbage_collect else 0
            defines["gpu_hash_dag_definitions.h"]["HASH_TABLE_ENABLE_GARBAGE_COLLECTION"] = 1
            defines["script_definitions.h"]["EDITS_COUNTERS"] = 1
            for replay, replay_depth in replays[scene]:
                profile_name = f"{scene}{2**(scene_depth-10)}k-{replay}-{hash_table}-memory"
                if auto_garbage_collect:
                    profile_name += "-gc"
                defines["script_definitions.h"]["REPLAY_NAME"] = escape_string(replay)
                defines["script_definitions.h"]["REPLAY_DEPTH"] = str(replay_depth)
                defines["profile_definitions.h"]["STATS_FILES_PREFIX"] = escape_string(profile_name)
                shared.compile_and_run_hash_dag(defines)"""
    

def benchmark_construction(scenes, use_material, profiling_folder):
    for scene, scene_depth in scenes:
        for hash_table in ["Atomic64", "TicketBoard", "AccelerationHash", "CompactAccelerationHash"]:
            defines = shared.default_script_definitions()
            defines["script_definitions.h"]["SCENE"] = escape_string(scene)
            defines["script_definitions.h"]["SCENE_DEPTH"] = scene_depth
            defines["script_definitions.h"]["EDITS_ENABLE_COLORS"] = 1 if use_material else 0
            defines["script_definitions.h"]["EDITS_ENABLE_MATERIALS"] = 1 if use_material else 0
            defines["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 1
            defines["script_definitions.h"]["DAG_TYPE"] = f"EDag::MyGpuDag"
            defines["script_definitions.h"]["SAVE_SCENE"] = 1
            defines["gpu_hash_dag_definitions.h"]["HASH_TABLE_TYPE"] = f"HashTableType::{hash_table}"
            defines["gpu_hash_dag_definitions.h"]["TARGET_LOAD_FACTOR"] = 96

            profile_name = f"{scene}{2**(scene_depth-10)}k-{hash_table}-construction"
            defines["profile_definitions.h"]["CONSTRUCT_STATS_FILE_PATH"] = escape_string(profile_name)
            defines["profile_definitions.h"]["PROFILING_PATH"] = escape_string(profiling_folder.replace("\\", "/"))
            shared.compile_and_run_hash_dag(defines)


def benchmark_path_tracing(scene, scene_depth, replays, material_bits, profiling_folder):
    for hash_table in ["Atomic64", "TicketBoard", "AccelerationHash", "CompactAccelerationHash", "HashDag"]:
        for path_trace_depth in range(2, 8):
            defines = shared.default_script_definitions()
            defines["script_definitions.h"]["SCENE"] = escape_string(scene)
            defines["script_definitions.h"]["SCENE_DEPTH"] = scene_depth
            defines["script_definitions.h"]["EDITS_ENABLE_COLORS"] = 1 if material_bits > 0 else 0
            defines["script_definitions.h"]["EDITS_ENABLE_MATERIALS"] = 1 if material_bits > 0 else 0
            defines["script_definitions.h"]["CAPTURE_GPU_TIMINGS"] = 1
            defines["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 0
            defines["script_definitions.h"]["DEFAULT_PATH_TRACE_DEPTH"] = path_trace_depth
            if hash_table ==  "HashDag":
                defines["script_definitions.h"]["DAG_TYPE"] = f"EDag::HashDag"
                defines["gpu_hash_dag_definitions.h"]["HASH_TABLE_TYPE"] = f"HashTableType::Atomic64"
            else:
                defines["script_definitions.h"]["DAG_TYPE"] = f"EDag::MyGpuDag"
                defines["gpu_hash_dag_definitions.h"]["HASH_TABLE_TYPE"] = f"HashTableType::{hash_table}"
            defines["gpu_hash_dag_definitions.h"]["TARGET_LOAD_FACTOR"] = 96
            defines["gpu_hash_dag_definitions.h"]["HASH_TABLE_ENABLE_GARBAGE_COLLECTION"] = 0
            defines["gpu_hash_dag_definitions.h"]["HASH_DAG_MATERIAL_BITS"] = material_bits
            defines["profile_definitions.h"]["PROFILING_PATH"] = escape_string(profiling_folder.replace("\\", "/"))
            construction_profile_name = f"{scene}{2**(scene_depth-10)}k-{hash_table}-construction"
            defines["profile_definitions.h"]["CONSTRUCT_STATS_FILE_PATH"] = escape_string(construction_profile_name)

            # Initial construction, saving the hash dag.
            if hash_table != "HashDag":
                defines["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 1
                defines["script_definitions.h"]["SAVE_SCENE"] = "1"
                shared.compile_and_run_hash_dag(defines)

            # Performance runs
            defines["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 0
            defines["script_definitions.h"]["SAVE_SCENE"] = "0"
            for replay, replay_depth in replays:
                defines["script_definitions.h"]["REPLAY_NAME"] = escape_string(replay)
                defines["script_definitions.h"]["REPLAY_DEPTH"] = str(replay_depth)
                for run in range(10):
                    profile_name = f"{scene}{2**(scene_depth-10)}k-{replay}-{hash_table}_run{run}"
                    defines["profile_definitions.h"]["STATS_FILES_PREFIX"] = escape_string(profile_name)
                    shared.compile_and_run_hash_dag(defines)

if __name__ == "__main__":
    scenes = [
        ("epiccitadel", 17),
        #("epiccitadel", 14),
        #("epiccitadel", 15),
        #("epiccitadel", 16),
        #("sanmiguel", 15),
        ("sanmiguel", 16),
    ]
    replays = {
        "epiccitadel": [("large_edits", 16), ("copy", 16)],
        "sanmiguel": [("tree_copy2", 16)]
    }

    load_build_tools()

    profiling_folder = os.path.join(shared.base_folder, "profiling", "office_windows_v7", "no_materials")
    shared.lazy_create_dir(profiling_folder)
    benchmark_path_tracing("epiccitadel", 17, [("pt_inside", 14)], 0, profiling_folder)

    profiling_folder = os.path.join(shared.base_folder, "profiling", "office_windows_v7", "materials4")
    shared.lazy_create_dir(profiling_folder)
    benchmark_path_tracing("epiccitadel", 17, [("pt_inside", 14)], 4, profiling_folder)
    

    #profiling_folder = os.path.join(shared.base_folder, "profiling", "linux_v7", "no_materials")
    #shared.lazy_create_dir(profiling_folder)
    #benchmark_methods(scenes, replays, 0, profiling_folder)

    #profiling_folder = os.path.join(shared.base_folder, "profiling", "linux_v7", "materials4")
    #shared.lazy_create_dir(profiling_folder)
    #benchmark_methods(scenes, replays, 4, profiling_folder)
