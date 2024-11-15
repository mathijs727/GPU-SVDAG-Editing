from shared import *
import pathlib
import os
import json
import xlsxwriter
import pandas
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pyperclip

plot_height = 2.3
plot_aspect = 3

def write_to_excel(rows, file_path):
    workbook = xlsxwriter.Workbook(file_path)
    worksheet = workbook.add_worksheet()
    for col, (key, values) in enumerate(rows.items()):
        worksheet.set_column(col, col, len(key) + 5)
        worksheet.write(0, col, key)
        for row, value in enumerate(values):
            worksheet.write(1 + row, col, value)
    workbook.close()


def convert_to_pandas(rows: list):
    return pandas.DataFrame(rows)


def scan_files(folder, filter=lambda _: True):
    rows = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        with open(file_path, "r") as f:
            if not filter(file_path):
                continue

            file_content = json.load(f)
            for hash_table in file_content:
                row = {}
                row["hash_table"] = hash_table["hash_table"]
                row["thread_type_add"] = hash_table["thread_type_add"]
                row["thread_type_find"] = hash_table["thread_type_find"]
                for define_files in hash_table["defines"].values():
                    for key, value in define_files .items():
                        row[f"setting_{key}"] = value
                settings = hash_table["settings"]
                for key, value in settings.items():
                    row[f"setting_{key}"] = value
                row["setting_loadFactor"] = np.rint(
                    (settings["numInitialItems"] + settings["numInsertItems"])
                    / settings["numBuckets"]
                ).astype(int)
                for key, value in hash_table["results"].items():
                    row[f"result_{key}"] = value
                rows.append(row)
    return rows


# https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
def format_bytes(size, max_unit=100000):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: "", 1: "Ki", 2: "Mi", 3: "Gi", 4: "Ti"}
    while size > power and n < max_unit:
        size /= power
        n += 1
    return f"{size:.1f} {power_labels[n]}B"

def format_bytes_in_GB(size):
    size_in_GB = size / (2**30)
    return f"{size_in_GB:.1f} GiB"

def format_items(num_items):
    if num_items < 1000:
        return f"{num_items} Items"
    elif num_items < 1000 * 1000:
        return f"{num_items//1000}K Items"
    else:
        return f"{num_items//1000//1000}M Items"


def check_not_mixing_measurements(df, whitelist=[]):
    # Make sure that we are not accidentaly averaging results of different settings
    for column in df:
        if column.startswith("setting_") and column not in whitelist:
            unique_settings = df[column].unique()
            if len(unique_settings) != 1:
                print(
                    f"Multiple values found for column {column}: {list(unique_settings)}"
                )
            assert len(unique_settings) == 1


def format_y_axis(ax, metric):
    if metric.endswith("_ms"):
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.1f} ms")
        ax.set_ylim(ymin=0)
    elif metric.endswith("_bytes"):
        ax.yaxis.set_major_formatter(lambda x, _: format_bytes_in_GB(x))
        ax.set_ylim(ymin=0)

def get_ui_setting_name(setting):
    lut = {"itemSizeInU32": "Item size (U32)", "loadFactor": "Load Factor"}
    return lut[setting]

def get_hash_table_ui_name(hash_table, thread_type_add=None, thread_type_find=None):
    lut = {
        "Atomic64HashTable": "Atomic U64",
        "AccelerationHashTable": "Acceleration Hash (32 bits)",
        "CompactAccelerationHashTable": "Acceleration Hash (8 bits)",
        "TicketBoardHashTable": "Ticket Board",
        "SlabHash": "SlabHash",
        "DyCuckoo": "DyCuckoo",
    }
    out = lut[hash_table]
    #if thread_type_add:
    #    out += f" {thread_type_add}/{thread_type_find}"
    return out


def get_ui_metric_name(setting):
    lut = {
        "bulkAdd1_ms": "Build",
        "bulkAdd2_ms": "Insert",
        "bulkSearch_hit25_ms": "Search (25%)",
        "bulkSearch_hit50_ms": "Search (50%)",
        "bulkSearch_hit75_ms": "Search (75%)",
        "insert1_memory_used_by_table_and_slabs_bytes": "Memory Used (Initial)",
        "insert2_memory_used_by_table_and_slabs_bytes": "Memory Used (After)",
        "insert1_memory_used_by_slabs_bytes": "Memory Used (Initial)",
        "insert2_memory_used_by_slabs_bytes": "Memory Used (After)",
        "insert1_memory_used_by_items_bytes": "Items Memory (Initial)",
        "insert2_memory_used_by_items_bytes": "Items Memory (After)",
        "insert1_memory_allocated_bytes": "Initial Memory ALLOCATED",
    }
    return lut[setting]


def format_table_value(metric, value):
    if metric.endswith("_ms"):
        return f"{value:.1f} ms"
    elif metric.endswith("_bytes"):
        return format_bytes(value, 2)
    else:
        return value

def make_latex_table(df, load_factor, methods, metrics):
    df = df[df["setting_loadFactor"] == load_factor]

    # l = left aligned; fit to content
    # X = column fills remaining width
    out = r"""\begin{table*}[]
\begin{tabularx}{\textwidth}{@{}lX"""
    out += (len(metrics) * "l")
    out += r"""@{}}
\toprule
"""
    out += " & ".join(["Item Size (U32)", "Method"] + [get_ui_metric_name(m).replace("%", "\\%") for m in metrics]) + "\\\\\n"
    
    for item_size_in_u32, row_df in df.groupby("setting_itemSizeInU32"):
        #if item_size_in_u32 < 2:
        #    continue

        out += "\\midrule\n"
        rows = []
        for hash_table, thread_add, thread_search in methods:
            method_df = row_df[(row_df["hash_table"] == hash_table) & (row_df["thread_type_add"] == thread_add) & (row_df["thread_type_find"] == thread_search)]
            if len(method_df) == 0:
                continue

            row = []
            row.append(item_size_in_u32 if len(rows) == 0 else "")
            row.append(get_hash_table_ui_name(hash_table))

            for metric in metrics:
                if metric.endswith("_bytes"):
                    tmp = method_df[method_df["setting_capture_memory_stats_slow"] == 1]
                else:
                    tmp = method_df[method_df["setting_capture_memory_stats_slow"] == 0]
                check_not_mixing_measurements(tmp)
                values = np.array(tmp[f"result_{metric}"])
                # Throw away top/bottom 2, which are considered outliers
                mean = values[0] if len(values) == 1 else np.mean(np.sort(values)[2:-2])
                #mean = tmp[f"result_{metric}"].mean()
                row.append(mean)
            rows.append(row)

        metric_lowest_scores = []
        for i, metric in enumerate(metrics):
            row_values = [r[i+2] for r in rows if r[i+2] != 0]
            metric_lowest_scores.append(np.min(row_values) if row_values else 0)
        
        def format_row(i, v):
            if v == 0:
                return "-"
            lowest = v == metric_lowest_scores[i]
            formatted_v = format_table_value(metrics[i], v)
            if lowest:
                return f"\\textbf{{{formatted_v}}}"
            else:
                return formatted_v

        for row in rows:
            out += f"{row[0]} & {row[1]} & "
            out += " & ".join([format_row(i, v) for i, v in enumerate(row[2:])])
            out += " \\\\\n"

    num_initial_items = df["setting_numInitialItems"].unique()[0]
    num_insert_items = df["setting_numInsertItems"].unique()[0]
    num_search_items = df["setting_numSearchItems"].unique()[0]
    caption = f"Results for searching {format_items(num_search_items)} and subsequently inserting {format_items(num_insert_items)} into a hash table initially storing {format_items(num_initial_items)} targeting a load factor (after insertion) of {load_factor}. Search performance is evaulated for various hit rates (number of search operations that succeed)."
    
    out += r"""\bottomrule
\end{tabularx}
\caption{""" + caption + r"""}
\label{table:hash_table_benchmark}
\end{table*}"""

    #print(out)

    print("\n\nTable copied to clipboard")
    pyperclip.copy(out)


def plot_seaborn(df, x, ys, filename=None):
    hash_table_order = {
        "Atomic64HashTable": 1,
        "TicketBoardHashTable": 2,
        "AccelerationHashTable": 3,
        "CompactAccelerationHashTable": 4,
    }
    thread_type_order = {
        "warp": 0,
        "warp_hybrid": 1
    }
    hash_table_types = list(zip(df["hash_table"], df["thread_type_add"], df["thread_type_find"]))
    hash_table_types_order = sorted(list(set(hash_table_types)), key=lambda tmp: 100*hash_table_order[tmp[0]]+10*thread_type_order[tmp[1]]+thread_type_order[tmp[2]])
    hash_table_order = [get_hash_table_ui_name(*types) for types in hash_table_types_order]
    df["hash_table_name"] = [get_hash_table_ui_name(*types) for types in hash_table_types]

    df.drop(columns=["hash_table", "thread_type_add", "thread_type_find"])

    sns.set_theme()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, axes = plt.subplots(len(ys), 1, figsize=(plot_height*plot_aspect, len(ys)*plot_height), sharex=True)
    for ax, metric in zip(axes, ys):
        metric_df = df[df["setting_capture_memory_stats_slow"] == metric.endswith("_bytes")]
        check_not_mixing_measurements(metric_df, [f"setting_{x}", "setting_numBuckets"])
        sns.barplot(
            ax=ax,
            data=metric_df,
            x=f"setting_{x}",
            y=f"result_{metric}",
            hue="hash_table_name",
            hue_order=hash_table_order
        )
        # Draw horizontal line at the memory used by the items
        if metric == "insert1_memory_used_by_slabs_bytes" or metric == "insert2_memory_used_by_slabs_bytes":
            items_memory_metric = "insert1_memory_used_by_items_bytes" if metric == "insert1_memory_used_by_slabs_bytes" else "insert2_memory_used_by_items_bytes"
            memory_used_by_items = metric_df[f"result_{items_memory_metric}"].unique()
            assert(len(memory_used_by_items) == 1)
            ax.axhline(y=memory_used_by_items, color="black")

            step_size = 0.5 * (1 << 30) # 500MiB
            num_steps = (metric_df[f"result_{metric}"].max() + step_size - 1) // step_size
            tmp = np.arange(0, num_steps) * step_size
            ax.set_yticks(tmp)
        ax.get_legend().remove()
        #ax.set_title(get_ui_metric_name(metric))
        ax.set_xlabel(get_ui_setting_name(x))
        ax.set_ylabel(get_ui_metric_name(metric))
        format_y_axis(ax, metric)

    fig.align_ylabels(axes)
    handles, labels = ax.get_legend_handles_labels()
    if filename:
        fig.legend(handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=2, fancybox=True, shadow=True, title="Method")
        plt.savefig(filename, bbox_inches="tight")
    else:
        fig.legend(handles, labels)
        plt.show()


if __name__ == "__main__":
    base_folder = pathlib.Path(__file__).parent.parent.resolve()
    profiling_folder = os.path.join(
        base_folder, "profiling", "linux_v7_1", "hash_tables", "load_factor"
    )
    df = convert_to_pandas(scan_files(profiling_folder))
    df = df[df["setting_hash_table_store_slabs_in_table"] == 1]
    
    methods = [
        ("SlabHash", "warp_hybrid", "warp_hybrid"),
        ("DyCuckoo", "warp_hybrid", "warp_hybrid"),
        ("Atomic64HashTable", "warp_hybrid", "warp_hybrid"),
        ("TicketBoardHashTable", "warp_hybrid", "warp_hybrid"),
        ("AccelerationHashTable", "warp_hybrid", "warp_hybrid"),
        ("CompactAccelerationHashTable", "warp_hybrid", "warp_hybrid"),
    ]
    #make_latex_table(df=df, load_factor=96, methods=methods, metrics=["bulkAdd2_ms", "bulkSearch_hit25_ms", "bulkSearch_hit50_ms", "bulkSearch_hit75_ms", "insert1_memory_used_by_slabs_bytes"])
    #exit(0)

    df = df[(df["thread_type_add"] == "warp_hybrid") & (df["thread_type_find"] == "warp_hybrid")]
    my_df = df[(df["hash_table"] != "DyCuckoo") & (df["hash_table"] != "DyCuckooStatic") & (df["hash_table"] != "SlabHash")]
    
    plot_seaborn(filename="hash_table_load_factor_size6.pdf", df=my_df[my_df["setting_itemSizeInU32"] == 6], x="loadFactor", ys=["bulkAdd2_ms", "bulkSearch_hit50_ms", "insert2_memory_used_by_slabs_bytes"])
    #plot_seaborn(df=my_df[my_df["setting_itemSizeInU32"] == 6], x="loadFactor", ys=["bulkAdd2_ms", "bulkSearch_hit50_ms", "insert2_memory_used_by_slabs_bytes"])
    #plot_seaborn(df=my_df[my_df["setting_itemSizeInU32"] == 4], x="loadFactor", ys=["bulkAdd2_ms", "bulkSearch_hit25_ms", "insert2_memory_used_by_slabs_bytes"], filename="test.png")
    #plot_seaborn(df[df["setting_loadFactor"] == 96], x="setting_loadFactor", ys=["bulkAdd2_ms", "bulkSearch_hit50_ms", "insert2_memory_used_by_slabs_bytes"], filename="")
