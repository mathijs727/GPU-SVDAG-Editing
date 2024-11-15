import os
import shutil
import pathlib
import difflib
import subprocess

base_folder = pathlib.Path(__file__).parent.parent.resolve()
vcpkg_base_folder = os.environ["VCPKG_ROOT"]

def escape_string(str):
    return f"\"{str}\""

def lazy_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def create_empty_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def default_script_definitions():
    defines = {
        "USE_REPLAY": "1",
        "DAG_TYPE": "EDag::MyGpuDag",
        "USE_BLOOM_FILTER": 1,
        "THREADED_EDITS": 1,
        "EDITS_COUNTERS": 0,
        "COPY_APPLY_TRANSFORM": 0,
        "COPY_CAN_APPLY_SWIRL": 0,
        "VERBOSE_EDIT_TIMES": 0,
        "COPY_WITHOUT_DECOMPRESSION": 0,
        "EDITS_ENABLE_COLORS": 0,
        "EDITS_ENABLE_MATERIALS": 0,
        "AUTO_GARBAGE_COLLECT": 0,
        "ENABLE_CHECKS": 0,
        "TRACK_GLOBAL_NEWDELETE": 0,
        "UNDO_REDO": 0,
        "COLOR_TREE_LEVELS": 8,
        "CAPTURE_GPU_TIMINGS": 0,
        "CAPTURE_MEMORY_STATS_SLOW": 0,
        "EDIT_PARALLEL_TREE_LEVELS": 8,
        "OPTIMIZE_FOR_BENCHMARK": 1,
        "EXIT_AFTER_REPLAY": 1,
    }
    profiling_defines = {
        "PROFILING_PATH": escape_string(""),
        "STATS_FILES_PREFIX": escape_string("dummy")
    }
    gpu_hash_dag_defines = {
        "TARGET_LOAD_FACTOR": 128,
        "HASH_TABLE_WARP_ADD": 1,
        "HASH_TABLE_WARP_FIND": 1,
        "HASH_TABLE_ENABLE_GARBAGE_COLLECTION": 1,
        "HASH_TABLE_ACCURATE_RESERVE_MEMORY": 1,
        "HASH_TABLE_TYPE": "HashTableType::Atomic64",
        "HASH_TABLE_HASH_METHOD": "HashMethod::SlabHashXor",
        "HASH_TABLE_STORE_SLABS_IN_TABLE": 1,
        "HASH_DAG_MATERIAL_BITS": 6,
    }

    return {
        "script_definitions.h": defines,
        "profile_definitions.h": profiling_defines,
        "gpu_hash_dag_definitions.h": gpu_hash_dag_defines
    }

def write_script_definitions(filename, defines):
    new_script_definitions = ""
    new_script_definitions += "#pragma once\n"
    new_script_definitions += "#include \"hash_dag_enum.h\"\n\n"
    for name, value in defines.items():
        new_script_definitions += f"#define {name} {value}\n"

    script_definitions_filepath = os.path.join(base_folder, "src", "configuration", filename)
    with open(script_definitions_filepath, "r") as f:
        old_script_definitions = f.read()
        if new_script_definitions == old_script_definitions:
            return
        else:
            for i,s in enumerate(difflib.ndiff(old_script_definitions, new_script_definitions)):
                if s[0]==' ': continue
                elif s[0]=='-':
                    print(u'Delete "{}" from position {}'.format(s[-1],i))
                elif s[0]=='+':
                    print(u'Add "{}" to position {}'.format(s[-1],i))    

        
    with open(script_definitions_filepath, "w") as f:
        f.write(new_script_definitions)
    with open(script_definitions_filepath, "r") as f:
        old_script_definitions = f.read()
        assert(old_script_definitions == new_script_definitions)

def compile_and_run_hash_dag(definition_files):
    for filename, defines in definition_files.items():
        write_script_definitions(filename, defines)
    
    build_folder = os.path.join(base_folder, "build_python")
    lazy_create_dir(build_folder)

    # "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    vcpkg_toolchain_file = os.path.join(vcpkg_base_folder, "scripts", "buildsystems", "vcpkg.cmake")
    subprocess.check_call(["cmake", "-GNinja", "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_VULKAN_MEMORY_ALLOCATOR=ON", f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain_file}", "../"], cwd=build_folder)
    subprocess.check_call(["ninja", "DAG_edits_demo"], cwd=build_folder)

    exe_name = "DAG_edits_demo.exe" if os.name == "nt" else "DAG_edits_demo"

    try:
        subprocess.check_call([os.path.join(build_folder, exe_name)], cwd=build_folder)
    except subprocess.CalledProcessError as e:
        print(f"CalledProcessError: {e}")

def load_build_tools():
    try:
        subprocess.check_call("ninja")
    except FileNotFoundError:
        tools_file = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
        print("Please execute the following command:")
        print(f"\"{tools_file}\"")
        exit(1)
    except subprocess.SubprocessError:
        pass # ninja: error: loading 'build.ninja': The system cannot find the file specified.


def compile_and_run_hash_benchmark(args, capture_memory_stats, store_slabs_in_table):
    definition_files = default_script_definitions()
    definition_files["script_definitions.h"]["CAPTURE_GPU_TIMINGS"] = 1
    definition_files["script_definitions.h"]["CAPTURE_MEMORY_STATS_SLOW"] = 1 if capture_memory_stats else 0
    definition_files["gpu_hash_dag_definitions.h"]["HASH_TABLE_HASH_METHOD"] = "HashMethod::SlabHashXor"
    definition_files["gpu_hash_dag_definitions.h"]["HASH_TABLE_STORE_SLABS_IN_TABLE"] = 1 if store_slabs_in_table else 0
    for filename, defines in definition_files.items():
        write_script_definitions(filename, defines)
    
    load_build_tools()
    build_folder = os.path.join(base_folder, "build_python")
    if not os.path.exists(build_folder):
        os.makedirs(build_folder)
        # "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        vcpkg_toolchain_file = os.path.join(vcpkg_base_folder, "scripts", "buildsystems", "vcpkg.cmake")
        subprocess.check_call(["cmake", "-GNinja", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain_file}", "../"], cwd=build_folder)
    subprocess.check_call(["ninja", "HashMapBench"], cwd=build_folder)
    try:
        executable = "HashMapBench" + (".exe" if os.name == "nt" else "")
        args_list = [i for k, v in args.items() for i in [k, str(v)]]
        subprocess.check_call([os.path.join(build_folder, "extras", "hashmap_bench", executable)] + args_list, cwd=build_folder)
    except subprocess.CalledProcessError as e:
        print(f"CalledProcessError: {e}")


def format_scene_size(v):
    return f"{v:.0f}MB"

def format_resolution(v):
    return str(2**v >> 10) + "K"

def format_scene_with_resolution(s, r):
    return f"{scene_names[s]} ({format_resolution(r)})"

def format_compression_time(v):
    return f"{v:.2f}s"

def format_compression_ratio(v):
    return f"{v*100:.2f}\\%"

def format_RMSE(v):
    return f"{v:.4f}"

def argmin(l):
    return l.index(min(l))

def format_bold(s):
    return f"\\textbf{{{s}}}"