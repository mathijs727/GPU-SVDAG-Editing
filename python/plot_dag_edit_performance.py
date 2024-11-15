import pandas as pd
import copy
import re
from shared import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib import ticker
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pathlib
import os
from collections import defaultdict
import json
import seaborn as sns
import time
from enum import Enum
import progressbar # pip install progressbar2
import pyperclip # pip install pyperclip

base_folder = pathlib.Path(__file__).parent.parent.resolve()

#plot_width = 5
#plot_aspect = 1.6
plot_width = 10
plot_aspect = 2.2
plot_height = plot_width / plot_aspect

class Unit(Enum):
    NONE = 1
    MEMORY = 2
    TIME = 3

def scan_files_to_pandas(folder, count_num_voxels=True):
    files = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".json")]
    stat_units = {}
    stat_devices = {}
    rows = []
    for file_path in progressbar.progressbar(files):
        with open(file_path, "r") as f:
            if not file_path.endswith("-construction.json"):
                file_content = json.load(f)
                settings = [(f"setting_{k}", v) for k, v in (list(file_content["gpu_hash_dag_definitions"].items()) + list(file_content["script_definitions"].items()))]
                # I forgot to add "DEFAULT_PATH_TRACE_DEPTH" to the settings before running the initial benchmarks.
                # This code recovers the trace depth from the file name.
                if "setting_DEFAULT_PATH_TRACE_DEPTH" not in dict(settings) and "tracedepth" in file_path:
                    file_name = os.path.basename(file_path)
                    trace_depth = int(re.search("tracedepth([0-9]+)", file_name).group(1))
                    settings.append(("setting_DEFAULT_PATH_TRACE_DEPTH", trace_depth))
                if "machine" in file_content:
                    settings = settings + list(file_content["machine"].items())
                for stats in file_content["stats"]:
                    frame = stats["frame"]
                    row = dict(settings)
                    row["file"] = file_path
                    row["frame"] = frame
                    for stat in stats["stats"]:
                        column_name = "result_" + stat["name"]
                        row[column_name] = stat["value"]

                        if column_name not in stat_devices:
                            stat_devices[column_name] = stat["device"]
                        else:
                            assert(stat_devices[column_name] == stat["device"])
                        if column_name not in stat_units:
                            stat_units[column_name] = stat["unit"]
                        else:
                            assert(stat_units[column_name] == stat["unit"])
                    
                    # Temporary because I made this change in-between benchmarks so it may be missing from some files.
                    # Should remove this line after re-running all benchmarks
                    if "setting_hash_dag_material_bits" in row:
                        del row["setting_hash_dag_material_bits"]
                    rows.append(row)
    
    df = pd.DataFrame(rows)

    if not count_num_voxels:
        return df

    # Collect the number of edited voxels per frame from the HashDag memory runs.
    tmp = df[df["setting_dag_type"] == "HashDag"]
    tmp = tmp[~df["result_num voxels"].isna()]
    tmp = tmp[tmp["setting_edits_counters"] == 1]
    tmp = tmp[tmp["setting_hash_table_enable_garbage_collection"] == 1]
    check_not_mixing_measurements(tmp, whitelist=["setting_scene", "setting_scene_depth", "setting_replay_name", "setting_replay_depth"])

    # Use original HashDag counters to get the number of edited voxels per frame
    lut = {}
    for i in progressbar.progressbar(range(len(tmp))):
        scene = tmp.iloc[i]["setting_scene"]
        scene_depth = tmp.iloc[i]["setting_scene_depth"]
        replay_name =  tmp.iloc[i]["setting_replay_name"]
        frame =  tmp.iloc[i]["frame"]
        voxel_count = int(tmp.iloc[i]["result_num voxels"])
        assert(voxel_count > 0)
        lut[(scene, scene_depth, replay_name, frame)] = voxel_count
    # Store the number of edited voxels for the GPU hash dags.
    df["result_voxel_count"] = 0
    for i, row in progressbar.progressbar(df.iterrows(), max_value=len(df.index)):
        scene = row["setting_scene"]
        scene_depth = row["setting_scene_depth"]
        replay_name = row["setting_replay_name"]
        frame = row["frame"]
        key = (scene, scene_depth, replay_name, frame)
        df.at[i, "result_voxel_count"] = lut.get(key, 0)
    df.drop(columns="result_num voxels")
    return df
    
def get_modified_ctime(f):
    return time.ctime(os.path.getmtime(f))

def cached_scan_files_to_pandas(folder, count_num_voxels=True):
    files = [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".json")]
    last_modified_date = max(files, key=get_modified_ctime)
    
    cache_file_path = os.path.join(folder, "cache.bin")
    if os.path.exists(cache_file_path) and get_modified_ctime(cache_file_path) > last_modified_date:
        return pd.read_pickle(cache_file_path)
    else:
        df = scan_files_to_pandas(folder, count_num_voxels=count_num_voxels)
        pd.to_pickle(df, cache_file_path)
        return df

# https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
def format_bytes(size, max_unit=100000, num_digits=1):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "Ki", 2: "Mi", 3: "Gi", 4: "Ti"}
    while size > power and n < max_unit:
        size /= power
        n += 1
    return f"{size:.{num_digits}f} {power_labels[n]}B"

def format_y_axis(ax, unit : Unit):
    if unit == Unit.TIME:
        ax.yaxis.set_major_formatter(lambda x, _: f"{x*1000:.0f} ms")
        ax.set_ylim(ymin=0)
    elif unit == Unit.MEMORY:
        ax.yaxis.set_major_formatter(lambda x, _: format_bytes(x))
        ax.set_ylim(ymin=0)

def get_axis_name(metric_name):
    lut = {
        "frame": "Frame Time (ms)",
        "total edits": "CPU Edit Time (ms)"
    }
    return lut[metric_name]

def get_method_name(dag_type, hash_table_type):
    if dag_type == "HashDag":
        return "HashDAG"
    else:
        lut = {
            "Atomic64": "Atomic U64",
            "AccelerationHash": "Acceleration Hash (32 bits)",
            "CompactAccelerationHash": "Acceleration Hash (8 bits)",
            "TicketBoard": "Ticket Board"
        }
        return lut[hash_table_type]

def get_method_order(filter_names=None):
    sorted_method_names = [
        get_method_name("MyGpuDag", "Atomic64"),
        get_method_name("MyGpuDag", "TicketBoard"),
        get_method_name("MyGpuDag", "AccelerationHash"),
        get_method_name("MyGpuDag", "CompactAccelerationHash"),
        get_method_name("HashDag", "Atomic64"),
    ]
    if filter_names is not None:
        sorted_method_names = [n for n in sorted_method_names if n in filter_names]
    return sorted_method_names

def get_ui_scene_name(scene_name, scene_depth):
    lut = {
        "epiccitadel": "Citadel",
        "sanmiguel": "San Miguel",
    }
    return f"{lut[scene_name]} {1<<(scene_depth-10)}K"

def get_ui_replay_name(replay_name):
    lut = {
        "copy": "Copy",
        "tree_copy2": "Copy",
        "large_edits": "Place Large Spheres",
    }
    return lut[replay_name]

def get_ui_metric_name(metric_name):
    lut = {
        "frame": "Frame Time",
        "total edits": "Edit without upload (CPU)",
        "upload_to_gpu": "Upload to GPU",
        "MyGPUHashDAG.memory_used_by_slabs": "Memory Used"
    }
    return lut.get(metric_name, metric_name)

def get_ui_x_name(x_name):
    lut = {
        "frame": "Frame",
        "result_voxel_count": "Voxel Count"
    }
    return lut[x_name]

def get_scene_order(df):
    preferred_order = [
        get_ui_scene_name("epiccitadel", 15),
        get_ui_scene_name("epiccitadel", 16),
        get_ui_scene_name("sanmiguel", 15),
        get_ui_scene_name("epiccitadel", 17)
    ]
    return [scene for scene in preferred_order if not df[df["scene_name"] == scene].empty]

def check_not_mixing_measurements(df, whitelist=[]):
     # Make sure that we are not accidentaly averaging results of different settings
    for column in df:
        if column.startswith("setting_") and column not in whitelist:
            unique_settings = df[column].unique()
            if len(unique_settings) > 1:
                print(f"Multiple values found for column {column}: {list(unique_settings)}")
                print(df["file"].unique())
            assert(len(unique_settings) <= 1)

def _different_plots_impl(filename, df, kind, x, y, y_unit : Unit, col, row=None, col_order=None,):
    if y_unit ==  Unit.MEMORY:
        df = df[df["setting_auto_garbage_collect"] == 1]
    else:
        df = df[df["setting_capture_memory_stats_slow"] == 0]

    # Replace settings_dag_Type & setting_hash_table_type by hash table name
    df["Method"] = [get_method_name(dag, table) for dag, table in zip(df["setting_dag_type"], df["setting_hash_table_type"])]
    df = df.drop(columns=["setting_dag_type", "setting_hash_table_type"])

    # Make sure we don't accidentaly mix results from tests with different settings
    check_not_mixing_measurements(df)

    sns.set_theme()
    kwargs = {}
    if kind == "line":
        kwargs["n_boot"] = 10
    if col_order:
        kwargs["col_order"] = col_order


    g = sns.relplot(data=df, kind=kind, x=x, y=f"result_{y}", col=col, row=row, hue="Method", style="Method",
                    style_order=get_method_order(df["Method"].unique()), hue_order=get_method_order(df["Method"].unique()),
                   height=plot_height, aspect=plot_aspect, legend=True, facet_kws={"sharex": False, "legend_out": False }, **kwargs)
    if col and row:
        g.set_titles("{col_name} ({row_name})")
    elif col:
        g.set_titles("{col_name}")
    else:
        g.set_titles("{row_name}")
    for ax in g.axes.flatten():
        ax.set(xlabel=get_ui_x_name(x), ylabel=get_ui_metric_name(y))
        # Limit graphs to 100ms to improve readability.
        if y_unit == Unit.TIME:
            ax.set_ylim(ymax = min(0.5, ax.get_ylim()[1]))

    format_y_axis(g.axes[0, 0], y_unit)
    if filename:
        sns.move_legend(g, "center left", bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=True)
        plt.tight_layout()
        #sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=True, fancybox=True, shadow=True, title="Method")

    # bbox_inches="tight" prevents legend from being cut off when it falls outside of the plot.
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()


def plot_different_scene_depths(df, scene, replay_name, kind, x, y, y_unit, filename=None):
    df = df[df["setting_scene"] == scene]
    df = df[df["setting_replay_name"] == replay_name]

    # Sort scenes by depth
    unique_scenes = list(set(zip(df["setting_scene"], df["setting_scene_depth"])))
    unique_scene_names = [get_ui_scene_name(*s) for s in unique_scenes]
    sort_indices = np.argsort([d for _, d in unique_scenes])
    sorted_scene_names = [unique_scene_names[i] for i in sort_indices]
    
    df["scene_name"] = [get_ui_scene_name(scene, depth) for scene, depth in zip(df["setting_scene"], df["setting_scene_depth"])]
    df = df.drop(columns=["setting_scene", "setting_scene_depth"])

    _different_plots_impl(filename, df, kind, x, y, y_unit, "scene_name", col_order=sorted_scene_names)


def plot_different_replays(df, scene, scene_depth, kind, x, y, y_unit, filename=None):
    df = df[df["setting_scene"] == scene]
    df = df[df["setting_scene_depth"] == scene_depth]

    # Sort scenes by depth
    df["replay_name"] = [get_ui_replay_name(replay_name) for replay_name in df["setting_replay_name"]]
    df = df.drop(columns=["setting_replay_name"])
    sorted_replay_names = sorted(df["replay_name"].unique())

    _different_plots_impl(filename, df, kind, x, y, y_unit, col="replay_name", col_order=sorted_replay_names)

def plot_different_scenes_and_replays(df, kind, x, y, y_unit, filename=None, filter_scenarios=None):
    if filter_scenarios:
        filter = np.zeros(len(df), dtype=bool)
        for scene, scene_depth, replay_name, use_material in filter_scenarios:
            filter |= ((df["setting_scene"] == scene) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name))
        df = df[filter]

    # Sort scenes by depth
    unique_scenes = list(set(zip(df["setting_scene"], df["setting_scene_depth"])))
    unique_scene_names = [get_ui_scene_name(*s) for s in unique_scenes]
    sort_indices = np.argsort([d for _, d in unique_scenes])
    sorted_scene_names = [unique_scene_names[i] for i in sort_indices]

    # Replace settings_dag_Type & setting_hash_table_type by hash table name
    df["scene_name"] = [get_ui_scene_name(scene, depth) for scene, depth in zip(df["setting_scene"], df["setting_scene_depth"])]
    df["replay_name"] = [get_ui_replay_name(replay_name) for replay_name in df["setting_replay_name"]]
    df = df.drop(columns=["setting_scene", "setting_scene_depth", "setting_replay_name"])
    _different_plots_impl(filename, df, kind, x, y, y_unit, col="scene_name", row="replay_name", col_order=sorted_scene_names)


def plot_different_scenarios(df, scenarios, kind, x, y, y_unit, filename=None):
    # Sort scenes by depth
    def get_scenario_name(scene, scene_depth, replay_name, use_materials=False):
        #assert(not use_materials)
        return f"{get_ui_scene_name(scene, scene_depth)} ({get_ui_replay_name(replay_name)})"
    
    df["scenario_name"] = [get_scenario_name(*args) for args in zip(df["setting_scene"], df["setting_scene_depth"], df["setting_replay_name"])]
    df = df.drop(columns=["setting_scene", "setting_scene_depth", "setting_replay_name"])

    print(df["scenario_name"].unique())
    ordered_scenario_names = [get_scenario_name(*args) for args in scenarios]
    df = df[np.isin(df, ordered_scenario_names).any(axis=1)]

    _different_plots_impl(filename, df, kind, x, y, y_unit, col="scenario_name", col_order=ordered_scenario_names)


def compute_mean_frame_render_time(df):
    tmp1 = df[df["metric_name"] == "paths"]
    tmp2 = df[df["metric_name"] == "colors"]
    tmp3 = df[df["metric_name"] == "shadows"]
    return tmp1["metric_value"].mean() + tmp2["metric_value"].mean() + tmp3["metric_value"].mean()


def plot_frame_breakdown_bar(df, scene, scene_depth, replay_name):
    df = df[(df["setting_scene"] == scene) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name)]
    #df = df[df["setting_dag_type"] == "HashDag"]
    df = df[df["setting_capture_memory_stats_slow"] == 0]
    # Replace settings_dag_Type & setting_hash_table_type by hash table name
    df["hash_table_name"] = [get_method_name(dag, table) for dag, table in zip(df["setting_dag_type"], df["setting_hash_table_type"])]
    df = df.drop(columns=["setting_dag_type", "setting_hash_table_type"])

    metrics = ["paths", "shadows", "colors"]
    num_frames = df["frame"].max()+1

    hash_tables = df["hash_table_name"].unique()
    n = len(hash_tables)
    bar_padding = 0.01
    group_margin = 0.1
    assert(n * bar_padding < 1)
    bar_width = (1 - 2 * group_margin - (n - 1) * bar_padding) / n

    colors = sns.color_palette(n_colors=len(metrics))

    sns.set_theme()
    fig, ax = plt.subplots(1, 1)
    frames = np.arange(num_frames)
    for i, (hash_table, hash_table_df) in enumerate(df.groupby("hash_table_name")):
        bottom = np.zeros(num_frames)
        for color, metric in zip(colors, metrics):
            # Make sure we don't accidentaly mix results from tests with different settings
            check_not_mixing_measurements(hash_table_df)

            bar_x = -0.5 + group_margin + frames + i * (bar_width + bar_padding) - 0.5 * bar_width
            metric_results = np.zeros(num_frames)
            for frame, frame_df in hash_table_df.groupby("frame"):
                metric_results[frame] = frame_df[f"result_{metric}"].mean()
            ax.bar(x=bar_x, width=bar_width, height=metric_results, bottom=bottom, color=color)
            bottom += metric_results

    ax.set_xticks(frames)
    ax.set_xlabel("Frame")
    format_y_axis(ax, Unit.TIME)
    ax.set_ylabel("Time")
    fig.legend()
    plt.show()


def plot_dag_compression(df, scenarios):
    df = df[df["setting_dag_type"] != "HashDag"]
    df = df[df["setting_capture_memory_stats_slow"] == 1]
    df = df[df["setting_hash_table_enable_garbage_collection"] == 1]
    df = df[~df["result_total edits"].isna()]
    df["result_dag_compression"] = df["result_edit_svo_size_bytes"] / df[f"result_edit_insert_size_bytes"]
    plot_different_scenes_and_replays(df, "scatter", "frame", "dag_compression", Unit.MEMORY, filter_scenarios=scenarios)


def plot_dag_compression2(df, scenarios):
    for (scene, scene_depth, replay_name, edits_enable_materials), tmp in df.groupby(["setting_scene", "setting_scene_depth", "setting_replay_name", "setting_edits_enable_materials"]):
        print(scene, scene_depth, replay_name, edits_enable_materials, len(tmp))
    df = df[df["setting_dag_type"] != "HashDag"]
    df = df[df["setting_capture_memory_stats_slow"] == 1]
    df = df[df["setting_auto_garbage_collect"] == 1]
    df = df[~df["result_total edits"].isna()]
    df["Compression Ratio"] = df[f"result_edit_insert_size_bytes"] / df["result_edit_svo_size_bytes"]
    

    filter = np.zeros(len(df), dtype=bool)
    for scene_name, scene_depth, replay_name, use_materials in scenarios:
        tmp = (df["setting_scene"] == scene_name) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name) & (df["setting_edits_enable_materials"] == use_materials)
        print(np.sum(tmp))
        filter |= (df["setting_scene"] == scene_name) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name) & (df["setting_edits_enable_materials"] == use_materials)
    df = df[filter]


    def get_scene_name(scene_name, scene_depth, replay_name, enable_materials):
        #title = f"{get_ui_scene_name(scene_name, scene_depth)} - {get_ui_replay_name(replay_name)}"
        title = f"{get_ui_scene_name(scene_name, scene_depth)}"
        if enable_materials:
            title += " (With Materials)"
        else:
            title += " (No Materials)"
        return title
    df["Scene Name"] = [get_scene_name(scene_name, scene_depth, replay_name, enable_materials) for scene_name, scene_depth, replay_name, enable_materials in zip(df["setting_scene"], df["setting_scene_depth"], df["setting_replay_name"], df["setting_edits_enable_materials"])]
    df = df.drop(columns=["setting_scene", "setting_scene_depth", "setting_replay_name"])

    #df = df[(df["setting_scene"] == scene) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name)]
    # Replace settings_dag_Type & setting_hash_table_type by hash table name
    #df["Method"] = [get_method_name(dag, table) for dag, table in zip(df["setting_dag_type"], df["setting_hash_table_type"])]
    #df = df.drop(columns=["setting_dag_type", "setting_hash_table_type"])

    check_not_mixing_measurements(df, whitelist=["setting_edits_enable_colors", "setting_edits_enable_materials", "setting_hash_table_enable_garbage_collection", "setting_edits_counters"])

    sns.set_theme()
    g = sns.relplot(df, x="frame", y="Compression Ratio", hue="Scene Name", kind="scatter", legend=True, height=plot_height, aspect=plot_aspect)
    ax = g.axes[0, 0]
    ax.yaxis.set_major_formatter(lambda x, _: f"{int(x*100)}%")
    ax.set_ylim(ymin=0, ymax=1)
    plt.show()



def plot_frame_breakdown_line(df, scene, scene_depth, replay_name, filename=None):
    df = df[(df["setting_scene"] == scene) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name)]
    df = df[df["setting_dag_type"] != "HashDag"]
    df = df[df["setting_capture_memory_stats_slow"] == 0]
    # Replace settings_dag_Type & setting_hash_table_type by hash table name
    df["hash_table_name"] = [get_method_name(dag, table) for dag, table in zip(df["setting_dag_type"], df["setting_hash_table_type"])]
    df = df.drop(columns=["setting_dag_type", "setting_hash_table_type"])

    accumulated_metrics = [
        ("GPU", "Construct SVO", ["createIntermediateSvoStructure_cuda setup", "createIntermediateSvoStructure_innerNode_cuda", "createIntermediateSvoStructure_leaf_cuda"]),
        ("GPU", "Remove SVO Duplicates", ["insertDuplicatesIntoHashTableAsWarp_kernel", "findUniqueInHashTableAsWarp_kernel1"]),
        ("GPU", "Merge with SVDAG", ["findUniqueInHashTableAsWarp_kernel2", "insertUniqueInHashTableAsWarp_kernel"]),
    ]
    absolute_metrics = [
        ("CPU", "Total Editing", ["total edits"])
    ]
    num_frames = df["frame"].max()+1

    #x_name = "result_voxel_count"
    x_name = "frame"
    x_ui_name = get_ui_x_name(x_name)

    df2 = []
    for _, row in df.iterrows():
        accumulation = 0
        for i, (ui_device, ui_metric, internal_metrics) in enumerate(accumulated_metrics):
            timing = np.sum([row[f"result_{metric}"] for metric in internal_metrics])
            df2.append({
                x_ui_name: row[x_name],
                "Method": row["hash_table_name"],
                "Metric": f"[{ui_device}] " + ("+" if i != 0 else " ") + ui_metric,
                "Time": accumulation + timing
            })
            accumulation += timing

        for ui_device, ui_metric, internal_metrics in absolute_metrics:
            timing = np.sum([row[f"result_{metric}"] for metric in internal_metrics])
            df2.append({
                x_ui_name: row[x_name],
                "Method": row["hash_table_name"],
                "Metric": f"[{ui_device}] {ui_metric}",
                "Time": timing
            })
    df2 = reversed(df2)
    df2 = pd.DataFrame(df2)
    sns.set_theme()
    unique_methods = df["hash_table_name"].unique()
    method_order = [m for m in get_method_order() if m in unique_methods]
    g = sns.relplot(df2, x=x_ui_name, y="Time", kind="line", hue="Method", hue_order=method_order, style="Metric", n_boot=10, legend=True, facet_kws={"legend_out": False}, height=plot_height, aspect=plot_aspect)
    ax = g.axes[0, 0]
    format_y_axis(ax, Unit.TIME)

    if filename:
        # https://stackoverflow.com/questions/56575756/how-to-split-seaborn-legend-into-multiple-columns
        h, l = ax.get_legend_handles_labels()
        ax.legend_.remove()
        g.fig.legend(h, l, "upper center", bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=True,  ncol=2)

    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()


def plot_edit_operation_bar(df, scene, scene_depth, replay_name, operation, filename=None):
    df = df[(df["setting_scene"] == scene) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name)]
    #df = df[df["setting_dag_type"] != "HashDag"]
    df = df[df["setting_capture_memory_stats_slow"] == 0]
    # Replace settings_dag_Type & setting_hash_table_type by hash table name
    df["hash_table_name"] = [get_method_name(dag, table) for dag, table in zip(df["setting_dag_type"], df["setting_hash_table_type"])]
    df = df.drop(columns=["setting_dag_type", "setting_hash_table_type"])

    metrics = {
        "Construct SVO": ["createIntermediateSvoStructure_cuda setup", "createIntermediateSvoStructure_innerNode_cuda", "createIntermediateSvoStructure_leaf_cuda"],
        "Remove SVO Duplicates": ["insertDuplicatesIntoHashTableAsWarp_kernel", "findUniqueInHashTableAsWarp_kernel1"],
        "Merge with SVDAG": ["findUniqueInHashTableAsWarp_kernel2", "insertUniqueInHashTableAsWarp_kernel"],
    }
    
    x_name = "frame"
    x_ui_name = get_ui_x_name(x_name)

    df2 = []
    for _, row in df.iterrows():
        if operation in metrics:
            timing = np.sum([row[f"result_{metric}"] for metric in metrics[operation]])
        else:
            timing = row[f"result_{operation}"]
        df2.append({
            x_ui_name: row[x_name],
            "Method": row["hash_table_name"],
            "Time": timing
        })
        
    df2 = pd.DataFrame(df2)
    sns.set_theme()
    unique_methods = df["hash_table_name"].unique()
    method_order = [m for m in get_method_order() if m in unique_methods]
    fig, ax = plt.subplots(1, 1, figsize=(plot_width, plot_height))
    fig.tight_layout()
    #sns.scatterplot(df2, x=x_ui_name, y="Time", hue="Method", hue_order=method_order, ax=ax, legend=True)
    sns.barplot(df2, x=x_ui_name, y="Time", hue="Method", hue_order=method_order, ax=ax)
    #g = sns.relplot(df2, x=x_ui_name, y="Time", kind="scatter", hue="Method", hue_order=method_order, legend=True, facet_kws={"legend_out": False}, height=plot_height, aspect=plot_aspect)
    #ax = g.axes[0, 0]
    format_y_axis(ax, Unit.TIME)
    #ax.set_ylim(ymax=5/1000)

    if filename:
        # https://stackoverflow.com/questions/56575756/how-to-split-seaborn-legend-into-multiple-columns
        h, l = ax.get_legend_handles_labels()
        ax.legend_.remove()
        fig.legend(h, l, "upper center", bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=True,  ncol=len(unique_methods))
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()




def nan_to_zero(v):
    if np.isnan(v):
        return 0
    else:
        return v

def plot_frame_breakdown_line2(ax, df, scene, scene_depth, replay_name, hide_hashdag, filename=None):
    df = df[(df["setting_scene"] == scene) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name)]
    df = df[df["setting_capture_memory_stats_slow"] == 0]
    # Replace settings_dag_Type & setting_hash_table_type by hash table name
    df["hash_table_name"] = [get_method_name(dag, table) for dag, table in zip(df["setting_dag_type"], df["setting_hash_table_type"])]
    df = df.drop(columns=["setting_dag_type", "setting_hash_table_type"])
    check_not_mixing_measurements(df)

    accumulated_metrics = [
        ("GPU", "Construct SVO", ["createIntermediateSvoStructure_cuda setup", "createIntermediateSvoStructure_innerNode_cuda", "createIntermediateSvoStructure_leaf_cuda"]),
        ("GPU", "Remove SVO Duplicates", ["insertDuplicatesIntoHashTableAsWarp_kernel", "findUniqueInHashTableAsWarp_kernel1"]),
        ("GPU", "Merge with SVDAG", ["findUniqueInHashTableAsWarp_kernel2", "insertUniqueInHashTableAsWarp_kernel", "updateChildPointers"]),
    ]
    absolute_metrics = [
        ("CPU", "Total Editing", ["total edits", "upload_to_gpu"])
    ]

    x_name = "frame"
    x_ui_name = get_ui_x_name(x_name)

    tmp = defaultdict(list)
    df2 = []
    for _, row in df.iterrows():
        accumulation = 0
        for i, (ui_device, ui_metric, internal_metrics) in enumerate(accumulated_metrics):
            timing = np.sum([row[f"result_{metric}"] for metric in internal_metrics])
            # NOTE(Mathijs): to hide the HashDAG we set the values to <0. Removing HashDAG from the dataframe is not possible however as it will change the colors of the other lines.
            if hide_hashdag and row["hash_table_name"] == "HashDAG":
                timing = -1000
            df2.append({
                x_ui_name: row[x_name],
                "Method": row["hash_table_name"],
                "Metric": f"[{ui_device}] " + ("+" if i != 0 else " ") + ui_metric,
                "Time": accumulation + timing
            })
            #if not np.isnan(timing):
            #    tmp[row["hash_table_name"] + " - " + ui_metric].append(timing)
            #for metric in internal_metrics:
            #    xtiming = row[f"result_{metric}"]
            #    if not np.isnan(xtiming):
            #        tmp[row["hash_table_name"] + " - " + metric].append(xtiming)
            accumulation += timing

        for ui_device, ui_metric, internal_metrics in absolute_metrics:
            timing = np.sum([nan_to_zero(row[f"result_{metric}"]) for metric in internal_metrics])
            # NOTE(Mathijs): to hide the HashDAG we set the values to <0. Removing HashDAG from the dataframe is not possible however as it will change the colors of the other lines.
            if hide_hashdag and row["hash_table_name"] == "HashDAG":
                timing = -1000
            df2.append({
                x_ui_name: row[x_name],
                "Method": row["hash_table_name"],
                "Metric": f"[{ui_device}] {ui_metric}",
                "Time": timing
            })
    df2 = reversed(df2)
    df2 = pd.DataFrame(df2)

    print(f"\n{scene}")
    for key, value in tmp.items():
        print(f"{key} = {np.mean(value)*1000:.3f}ms")

    unique_methods = df["hash_table_name"].unique()
    method_order = [m for m in get_method_order() if m in unique_methods]
    #metric_order = [f"{m[0]} {m[1]}" for m in absolute_metrics] + [f"{m[0]} {m[1]}" for m in accumulated_metrics]
    sns.lineplot(df2, x=x_ui_name, y="Time", hue="Method", hue_order=method_order, style="Metric", n_boot=10, ax=ax, legend=True)
    format_y_axis(ax, Unit.TIME)


def make_big_frame_graph_plot(df, df_materials, filename=None):
    # scene, scene_depth, replay_name, use_materials
    scenarios = [
        ("sanmiguel", 16, "tree_copy2", True),
        ("epiccitadel", 17, "large_edits", False),
        ("epiccitadel", 17, "large_edits", True),
    ]
    num_scenarios = len(scenarios)

    sns.set_theme()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, axis = plt.subplots(1, num_scenarios, figsize=(num_scenarios * plot_width, plot_height), sharey=True)
    if not isinstance(axis, list):
        axis = [axis]
    for ax, (scene, scene_depth, replay_name, use_materials) in zip(axis, scenarios):
        selected_df = df_materials if use_materials else df
        plot_frame_breakdown_line2(ax=ax, df=selected_df , scene=scene, scene_depth=scene_depth, replay_name=replay_name, hide_hashdag=False)

        title = get_ui_scene_name(scene, scene_depth)
        if use_materials:
            title += " (4-bit Materials)"
        else:
            title += " (No Materials)"
        #ax.set_title(title)

    handles, labels = axis[0].get_legend_handles_labels()
    for ax in axis:
        ax.set_ylim(0, 0.008)
        ax.get_legend().remove()

    if filename:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fancybox=True, shadow=True)
        fig.tight_layout()
        plt.savefig(filename, bbox_inches="tight")
    else:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.9, 0.9), fancybox=True, shadow=True)
        fig.tight_layout()
        plt.show()


def make_memory_usage_table(no_mat_profiling_folder, mat4_profiling_folder, scenarios):
    scenes = [(scene, depth, use_materials) for scene, depth, _, use_materials in scenarios]

    # l = left aligned; fit to content
    # X = column fills remaining width
    out = r"""\begin{table}[]
\begin{tabularx}{\columnwidth}{@{}lXl@{}}
\toprule
Scene & Method & Memory\\
"""
    for scene, scene_depth, use_materials in scenes:
        scene_name = get_ui_scene_name(scene, scene_depth)
        if use_materials:
            material_name = " 4-bit Materials"
        else:
            material_name = " No Materials"
        
        out += "\\midrule\n"
        results = []
        items_memory_usage = 0
        for table in ["Atomic64", "TicketBoard", "AccelerationHash", "CompactAccelerationHash"]:
            file_name = f"{scene}{1<<(scene_depth-10)}k-{table}-construction.json"
            file_path = os.path.join(mat4_profiling_folder if use_materials else no_mat_profiling_folder, file_name)
            assert(os.path.exists(file_path))
            with open(file_path, "r") as f:
                j = json.load(f)
                memory = j["stats"]["svdag_memory_used_by_slabs"]
                items_memory_usage = j["stats"]["svdag_node_memory_in_bytes"] + j["stats"]["svdag_leaf_memory_in_bytes"]
            hash_table_name = get_method_name("MyGpuDag", table)
            results.append((hash_table_name, memory))
        
        min_memory = np.min([memory for _, memory in results])
        results = [("Nodes/Leaves Only", items_memory_usage)] + results
        for i, (table, memory) in enumerate(results):
            if i == 0:
                out += scene_name
            elif i == 1:
                out += material_name
            out += " & "
            out += table + " & "
            memory_str = format_bytes(memory, max_unit=2, num_digits=0)
            if memory == min_memory:
                out += f"\\textbf{{{memory_str}}} \\\\\n"
            else:
                out += f"{memory_str} \\\\\n"

    caption = "Memory usage of the tested scenes both with (4-bit) and without (N/A) materials. This includes memory that is allocated but not currently used (partially filled slabs)."
    
    out += r"""\bottomrule
\end{tabularx}
\caption{""" + caption + r"""}
\label{table:scene_memory}
\end{table}"""

    #print(out)

    print("\n\nTable copied to clipboard")
    pyperclip.copy(out)


def print_hashdag_stats(df, scene, scene_depth, replay_name):
    df = df[(df["setting_scene"] == scene) & (df["setting_scene_depth"] == scene_depth) & (df["setting_replay_name"] == replay_name)]
    df = df[df["setting_dag_type"] == "HashDag"]
    df = df[df["setting_capture_memory_stats_slow"] == 0]
    check_not_mixing_measurements(df)

    edit_time = df["result_total edits"].mean()
    upload_time = df["result_upload_to_gpu"].mean()
    total_edit_time = edit_time + upload_time
    print(f"edit_time = {edit_time}")
    print(f"upload_time = {upload_time}")
    print(f"total_edit_time = {total_edit_time}")


def plot_path_trace_depth_performance(df, scene, scene_depth, replay_name, filename=None):
    sns.set_theme()
    fig, axis = plt.subplots(1, 1, figsize=(plot_width, plot_height))

    df = df[df["setting_scene"] == scene]
    df = df[df["setting_scene_depth"] == scene_depth]
    df = df[df["setting_replay_name"] == replay_name]
    df = df[df["setting_dag_type"] == "MyGpuDag"]

    plots = []
    for hash_table_name, hash_table_df in df.groupby("setting_hash_table_type"):
        x = []
        y = []
        for path_trace_depth, path_trace_depth_df in hash_table_df.groupby("setting_DEFAULT_PATH_TRACE_DEPTH"):
            path_trace_depth_df = path_trace_depth_df[path_trace_depth_df["result_path_tracing"] > 0]
            path_trace_time_in_ms = path_trace_depth_df["result_path_tracing"].mean() * 1000
            x.append(path_trace_depth)
            y.append(path_trace_time_in_ms)
        sorting = np.argsort(x)
        x = np.array(x)[sorting]
        y = np.array(y)[sorting]
        check_not_mixing_measurements(path_trace_depth_df)
        #plots.append((hash_table_name, lambda x=x,y=y,hash_table_name=hash_table_name: axis.plot(x, y, label=hash_table_name)))
        plots.append((get_method_name("MyGpuDag", hash_table_name), x, y))

        if hash_table_name == "Atomic64":
            atomic_y = y
        elif hash_table_name == "AccelerationHash":
            acceleration_y = y

    methods_order = get_method_order([name for name, _, _ in plots])
    plots = sorted(plots, key = lambda x: methods_order.index(x[0]))
    for name, x, y in plots:
        axis.plot(x, y, label=name)

    print(f"diff: {list(zip(x, atomic_y / acceleration_y))}")

    axis.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:g} ms"))
    axis.set_xlabel("Path Depth")
    axis.set_ylabel("Render Time (ms)")
    # https://stackoverflow.com/questions/56575756/how-to-split-seaborn-legend-into-multiple-columns
    h, l = axis.get_legend_handles_labels()
    fig.legend(h, l, "upper center", bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=True,  ncol=4)
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    no_mat_profiling_folder = os.path.join(base_folder, "profiling", "linux_v7_1", "no_materials")
    df = cached_scan_files_to_pandas(no_mat_profiling_folder)
    df_no_hashdag = df [df ["setting_dag_type"] != "HashDag"]

    mat4_profiling_folder = os.path.join(base_folder, "profiling", "linux_v7_1", "materials4")
    df_materials = cached_scan_files_to_pandas(mat4_profiling_folder)

    if False:
        tmp = df
        tmp = tmp[tmp["setting_scene"] == "epiccitadel"]
        tmp = tmp[tmp["setting_scene_depth"] == 17]
        tmp = tmp[tmp["setting_replay_name"] == "large_edits"]
        tmp = tmp[tmp["setting_capture_memory_stats_slow"] == 0]
        tmp_hashdag = tmp[tmp["setting_dag_type"] == "HashDag"]
        tmp_gpudag = tmp[(tmp["setting_dag_type"] == "MyGpuDag") & (tmp["setting_hash_table_type"] == "Atomic64")]
        check_not_mixing_measurements(tmp_hashdag)
        check_not_mixing_measurements(tmp_gpudag)
        
        print("HashDAG mean render time: " + str(compute_mean_frame_render_time(tmp_hashdag)))
        print("GPU DAG mean render time: " + str(compute_mean_frame_render_time(tmp_gpudag)))

        print("HashDAG mean frame time: " + str(tmp_hashdag[tmp_hashdag["metric_name"] == "frame"]["metric_value"].mean()))
        print("GPU DAG mean frame time: " + str(tmp_gpudag[tmp_gpudag["metric_name"] == "frame"]["metric_value"].mean()))
        exit(0)

    scenarios = (
        ("sanmiguel", 16, "tree_copy2", True),
        ("epiccitadel", 17, "large_edits", False),
        ("epiccitadel", 17, "large_edits", True),
        #("epiccitadel", 17, "copy"),
    )
    xxx = pathlib.Path("C:/Users/mathi/OneDrive - Delft University of Technology/TU Delft/Projects/GPU DAG Editing/conference")
    #make_memory_usage_table(no_mat_profiling_folder, mat4_profiling_folder, scenarios)
    combined_df = pd.concat([df, df_materials])
    #plot_dag_compression2(combined_df[combined_df["setting_hash_table_type"] == "Atomic64"], scenarios)
    #print_hashdag_stats(df, "epiccitadel", 17, "large_edits")
    #make_big_frame_graph_plot(filename="combined_frame_graph_breakdown.pdf", df=df[df["setting_dag_type"] != "HashDag"], df_materials=df_materials[df_materials["setting_dag_type"] != "HashDag"])
    #make_big_frame_graph_plot(filename=xxx/"result_sanmiguel_mat4.png", df=df[df["setting_dag_type"] != "HashDag"], df_materials=df_materials[df_materials["setting_dag_type"] != "HashDag"])
    #exit(0)

    #plot_different_scenarios(filename=xxx/"result_hashdag_sanmiguel_nomat.png", df=df, scenarios=[("sanmiguel", 16, "tree_copy2", False)], kind="line", x="frame", y="MyGPUHashDAG.memory_used_by_items", y_unit=Unit.MEMORY)
    #plot_different_scenarios(filename=xxx/"result_mem_citadel_mat4.png", df=df_materials, scenarios=[("epiccitadel", 17, "large_edits", True)], kind="line", x="frame", y="MyGPUHashDAG.memory_used_by_slabs", y_unit=Unit.MEMORY)
    #plot_different_scenarios(filename="scenarios_frame_time.png", df=df, scenarios=scenarios, kind="line", x="frame", y="frame", y_unit=Unit.TIME)
    #plot_different_replays(filename="replays_citadel17_frame_time.pdf", df=df, scene="epiccitadel", scene_depth=17, kind="line", x="frame", y="frame", y_unit=Unit.TIME)
    #plot_different_scene_depths(filename="large_edits_frame_time.pdf", df=df, scene="epiccitadel", replay_name="large_edits", kind="line", x="frame", y="frame", y_unit=Unit.TIME)
    #plot_different_scenes_and_replays(filename="voxel_counts.pdf", df=df, kind="line", x="result_voxel_count", y="frame", y_unit=Unit.TIME)
    #plot_different_scenes_and_replays(df, kind="line", x="result_voxel_count", y="frame", y_unit=Unit.TIME)
    #plot_frame_breakdown_line(filename="edit_breakdown.pdf", df=df, scene="epiccitadel", scene_depth=17, replay_name="large_edits")
    #plot_frame_breakdown_line(df=df, scene="epiccitadel", scene_depth=17, replay_name="large_edits")
    #plot_edit_operation_bar(filename="result_citadel17_nomat_merge_with_svdag.png", df=df, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="Merge with SVDAG")

    if True: # PG2024 Presentation
        #plot_edit_operation_bar(filename="result_citadel17_nomat_frame.png", df=df, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="frame")
        #plot_edit_operation_bar(filename="result_citadel17_nomat_construct_svo.png", df=df_no_hashdag, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="Construct SVO")
        #plot_edit_operation_bar(filename="result_citadel17_nomat_remove_svo_duplicates.png", df=df_no_hashdag, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="Remove SVO Duplicates")
        #plot_edit_operation_bar(filename="result_citadel17_nomat_merge_with_svdag.png", df=df_no_hashdag, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="Merge with SVDAG")

        #plot_edit_operation_bar(filename="result_citadel17_mat4_frame.png", df=df_materials, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="frame")
        #plot_edit_operation_bar(filename="result_citadel17_mat4_nohashdag_frame.png", df=df_materials[df_materials["setting_dag_type"] != "HashDag"], scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="frame")
        #plot_edit_operation_bar(filename="result_citadel17_mat4_construct_svo.png", df=df_materials, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="Construct SVO")
        #plot_edit_operation_bar(filename="result_citadel17_mat4_remove_svo_duplicates.png", df=df_materials, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="Remove SVO Duplicates")
        #plot_edit_operation_bar(filename="result_citadel17_mat4_merge_with_svdag.png", df=df_materials, scene="epiccitadel", scene_depth=17, replay_name="large_edits", operation="Merge with SVDAG")

        no_mat_profiling_folder = os.path.join(base_folder, "profiling", "linux_v7_pt_perf", "no_materials")
        df = cached_scan_files_to_pandas(no_mat_profiling_folder, count_num_voxels = False)
        plot_path_trace_depth_performance(filename="result_my_bunny14_pt_inside.png", df=df, scene="my_bunny", scene_depth=14, replay_name="pt_inside")
        #plot_path_trace_depth_performance(df=df, scene="my_bunny", scene_depth=14, replay_name="pt_inside")
    exit(0)

    df_without_hashdag = df[df["setting_dag_type"] != "HashDag"]
    df_without_hashdag = df_without_hashdag[df_without_hashdag["setting_hash_table_enable_garbage_collection"] == 1]
    plot_different_scenarios(filename="edit_memory_usage.pdf", df=df_without_hashdag, scenarios=scenarios, kind="line", x="frame", y="MyGPUHashDAG.memory_used_by_slabs", y_unit=Unit.MEMORY)