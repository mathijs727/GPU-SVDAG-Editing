import os
import shared


def benchmark_warps_per_workgroup():
    profiling_folder = os.path.join(shared.base_folder, "profiling", "linux_v7", "hash_tables_slab_in_table")
    shared.lazy_create_dir(profiling_folder)
    
    #warps_per_work_group = 2
    #max_num_work_groups = 16 * 1024

    #initial_size_in_bytes = 512 * 1024 * 1024
    #insert_size_in_bytes = 128 * 1024 * 1024
    #search_size_in_bytes = 128 * 1024 * 1024
    num_initial_items = 32 * 1024 * 1024
    num_insert_items = 8 * 1024 * 1024
    num_search_items = 8 * 1024 * 1024
    reserved_overhead = 1.5 # How much memory is reserved (measured in items) relative ot the number of items inserted.

    load_factor = 128

    for item_size_in_u32 in range(1, 10):
        #item_size_in_bytes = item_size_in_u32 * 4
        #num_initial_items = initial_size_in_bytes // item_size_in_bytes
        #num_insert_items = insert_size_in_bytes // item_size_in_bytes
        num_total_items = num_initial_items + num_insert_items
        num_reserved_items = int(num_total_items * reserved_overhead)
        #num_search_items = search_size_in_bytes // item_size_in_bytes

        num_buckets = int(num_total_items / load_factor) + 1
        
        for warps_per_work_group in [2]:
            for max_num_work_groups in [16384]:
                out_file_name = f"item_size_{item_size_in_u32}_warps_{warps_per_work_group}_workgroups_{max_num_work_groups}_run_{run}.json"
                out_file_path = os.path.join(profiling_folder, out_file_name)
                args = {
                    "--item_size": item_size_in_u32,
                    "--num_initial_items": num_initial_items,
                    "--num_insert_items": num_insert_items,
                    "--num_reserved_items" : num_reserved_items,
                    "--num_search_items" : num_search_items,
                    "--num_buckets": num_buckets,
                    "--warps_per_work_group" : warps_per_work_group,
                    "--max_num_work_groups": max_num_work_groups,
                    "--runs": "10",
                    "--out": out_file_path,
                }
                shared.compile_and_run_hash_benchmark(args, False)


def benchmark_load_factor():
    profiling_folder = os.path.join(shared.base_folder, "profiling", "linux_v7", "hash_tables", "load_factor")
    shared.lazy_create_dir(profiling_folder)
    
    warps_per_work_group = 2
    max_num_work_groups = 16 * 1024

    #initial_size_in_bytes = 512 * 1024 * 1024
    #insert_size_in_bytes = 128 * 1024 * 1024
    #search_size_in_bytes = 128 * 1024 * 1024
    num_initial_items = 32 * 1024 * 1024
    num_insert_items = 8 * 1024 * 1024
    num_search_items = 8 * 1024 * 1024
    reserved_overhead = 1.5 # How much memory is reserved (measured in items) relative ot the number of items inserted.

    for store_slabs_in_table in [True, False]:
        for capture_memory_stats in [True, False]:
            for item_size_in_u32 in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                #item_size_in_bytes = item_size_in_u32 * 4
                #num_initial_items = initial_size_in_bytes // item_size_in_bytes
                #num_insert_items = insert_size_in_bytes // item_size_in_bytes
                num_total_items = num_initial_items + num_insert_items
                num_reserved_items = int(num_total_items * reserved_overhead)
                #num_search_items = search_size_in_bytes // item_size_in_bytes


                for load_factor in [32, 48, 64, 80, 96, 112, 128]:
                    num_buckets = int(num_total_items / load_factor) + 1
                    args = {
                        "--item_size": item_size_in_u32,
                        "--num_initial_items": num_initial_items,
                        "--num_insert_items": num_insert_items,
                        "--num_reserved_items" : num_reserved_items,
                        "--num_search_items" : num_search_items,
                        "--num_buckets": num_buckets,
                        "--warps_per_work_group" : warps_per_work_group,
                        "--max_num_work_groups": max_num_work_groups
                    }
                    #out_file_name = f"item_size_{item_size_in_u32}_load_factor_{load_factor}_run_{run}.json"
                    def get_out_file_path(postfix):
                        filename = f"item_size_{item_size_in_u32}_lf_{load_factor}_{postfix}.json"
                        return os.path.join(profiling_folder, filename)

                    postfix = ""
                    if capture_memory_stats:
                        postfix += "mem_"
                        args["--runs"] = 1
                    else:
                        args["--runs"] = 10
                    
                    if not store_slabs_in_table:
                        postfix += "not"
                    postfix += "slabintable"

                    args["--out"] = get_out_file_path(postfix)

                    shared.compile_and_run_hash_benchmark(args, capture_memory_stats=capture_memory_stats, store_slabs_in_table=store_slabs_in_table)



if __name__ == "__main__":
    benchmark_load_factor()