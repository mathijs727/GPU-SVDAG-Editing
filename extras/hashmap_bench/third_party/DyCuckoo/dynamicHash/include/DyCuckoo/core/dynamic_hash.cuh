#ifndef DYNAMIC_HASH_H
#define DYNAMIC_HASH_H
#include "DyCuckoo/data/data_layout.cuh"
#include "cnmem.h"
#include <cooperative_groups.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>
using namespace cooperative_groups;
using namespace cuckoo_helpers;
using namespace hashers;

namespace cg = cooperative_groups;
namespace ch = cuckoo_helpers;

namespace cuckoo_helpers {
DEVICEQUALIFIER uint32_t atomicCAS_linux_dumb_dumb(uint32_t* address, uint32_t compare, uint32_t val)
{
    return atomicCAS(address, compare, val);
}
DEVICEQUALIFIER uint64_t atomicCAS_linux_dumb_dumb(uint64_t* address, uint64_t compare, uint64_t val)
{
    return atomicCAS((unsigned long long*)address, compare, val);
}

DEVICEQUALIFIER uint32_t atomicExch_linux_dumb_dumb(uint32_t* address, uint32_t val)
{
    return atomicExch(address, val);
}
DEVICEQUALIFIER uint64_t atomicExch_linux_dumb_dumb(uint64_t* address, uint64_t val)
{
    return atomicExch((unsigned long long*)address, val);
}

}

template <typename MyDataLayout>
struct DynamicHash {
    MyDataLayout::cuckoo_t cuckoo_table;
    MyDataLayout::error_table_t error_table;

    using key_t = typename MyDataLayout::key_t;
    using bucket_t = typename MyDataLayout::bucket_t;
    using cuckoo_t = typename MyDataLayout::cuckoo_t;
    using error_table_t = typename MyDataLayout::error_table_t;

    static constexpr uint8_t CgSize = 16;
    static constexpr uint8_t MaxEvictNum = 100;

    static constexpr key_t empty_key = MyDataLayout::empty_key;
    static constexpr uint32_t bucket_size = MyDataLayout::bucket_size;
    static constexpr uint32_t error_table_len = MyDataLayout::error_table_len;

    static constexpr uint8_t cg_size = CgSize;
    static constexpr uint8_t max_evict_num = MaxEvictNum;

    DEVICEQUALIFIER
    void cg_error_handle(key_t& data, thread_block_tile<cg_size> group)
    {
        if (group.thread_rank() != 0)
            return;
        uint32_t ptr = atomicAdd(&(error_table.error_pt), 1);
        if (ptr >= error_table_len) {
            return;
        }
        error_table.error_keys[ptr] = data;
    }

    DEVICEQUALIFIER INLINEQUALIFIER bool cg_inert(key_t& data, uint8_t pre_table_no, thread_block_tile<cg_size> group)
    {
        auto lane_id = group.thread_rank();
        key_t key;
        uint32_t pair;
        uint32_t insert_table_no;
        for (uint32_t i = 0; i < MaxEvictNum; ++i) {
            key = data;
            pair = ch::get_pair((uint32_t)key);
            insert_table_no = ch::get_table1_no(pair);
            if (insert_table_no == pre_table_no) {
                insert_table_no = ch::get_table2_no(pair);
            }
            uint32_t table_len = cuckoo_table.table_size[insert_table_no];
            uint32_t hash_val = ch::caculate_hash((uint32_t)key, insert_table_no, table_len);
            bucket_t* bucket = cuckoo_table.table_group[insert_table_no] + hash_val;
            uint32_t stride = bucket_size / cg_size;
            for (uint32_t ptr = 0; ptr < stride; ptr++) {
                key_t probe_data = (bucket->bucket_data)[ptr * cg_size + lane_id];
                key_t probe_key = probe_data;
                uint32_t group_mask = group.ballot(probe_key == empty_key);
                bool success = false;
                while (group_mask != 0) {
                    uint32_t group_leader = __ffs(group_mask) - 1;
                    if (lane_id == group_leader) {
                        auto result = cuckoo_helpers::atomicCAS_linux_dumb_dumb((key_t*)(bucket->bucket_data) + ptr * cg_size + lane_id, probe_data, data);
                        if (result == probe_data) {
                            success = true;
                        }
                    }
                    if (group.any(success == true)) {
                        return true;
                    } else {
                        probe_data = (bucket->bucket_data)[ptr * cg_size + lane_id];
                        probe_key = probe_data;
                        group_mask = group.ballot(probe_key == empty_key);
                    }
                }
            }
            // probe fail, evict
            key_t cas_result;
            if (lane_id == 0) {
                cas_result = cuckoo_helpers::atomicExch_linux_dumb_dumb((key_t*)(bucket->bucket_data) + (i % cg_size), data);
            }
            data = group.shfl(cas_result, 0);
            pre_table_no = insert_table_no;
        }
        // insert fail, handle error
        cg_error_handle(data, group);
        return false;
    }

    DEVICEQUALIFIER
    void cuckoo_insert(key_t* keys, uint32_t data_num)
    {
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;

        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);

        for (; group_index_in_all < data_num; group_index_in_all += step) {
            uint32_t pre_table_no;
            key_t key = keys[group_index_in_all];
            uint32_t pair = ch::get_pair((uint32_t)key);
            if (key & 1) {
                /// last bit == 1 , first insert to pos 2
                pre_table_no = ch::get_table1_no(pair);
            } else {
                pre_table_no = ch::get_table2_no(pair);
            }
            cg_inert(key, pre_table_no, group);
        }
    }

    DEVICEQUALIFIER INLINEQUALIFIER
        uint8_t
        cg_search_in_bucket(const key_t& key, uint32_t& value, bucket_t* bucket, thread_block_tile<cg_size> group)
    {
        auto lane_id = group.thread_rank();
        uint32_t stride = bucket_size / cg_size;
        for (uint32_t ptr = 0; ptr < stride; ptr++) {
            key_t probe_data = (bucket->bucket_data)[ptr * cg_size + lane_id];
            key_t probe_key = probe_data;
            uint32_t group_mask = group.ballot(probe_key == key);
            if (group_mask != 0) {
                uint32_t group_leader = __ffs(group_mask) - 1;
                if (lane_id == group_leader) {
                    value = 1;
                }
                return true;
            }
        }
        value = 0xFFFFFFFF;
        return false;
    }

    DEVICEQUALIFIER
    void cuckoo_search(key_t* keys, uint32_t* values, uint32_t size)
    {
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);

        uint32_t pair, search_table_no, table_len, hash_val;
        bucket_t* bucket;
        uint8_t flag = false;
        for (; group_index_in_all < size; group_index_in_all += step) {
            key_t key = keys[group_index_in_all];
            pair = ch::get_pair((uint32_t)key);
            if (key & 1) {
                search_table_no = ch::get_table2_no(pair);
            } else {
                search_table_no = ch::get_table1_no(pair);
            }
            table_len = cuckoo_table.table_size[search_table_no];
            hash_val = ch::caculate_hash((uint32_t)key, search_table_no, table_len);
            bucket = cuckoo_table.table_group[search_table_no] + hash_val;
            flag = cg_search_in_bucket(key, values[group_index_in_all], bucket, group);
            if (group.any(flag == true)) {
                continue;
            }
            if (key & 1) {
                search_table_no = ch::get_table1_no(pair);
            } else {
                search_table_no = ch::get_table2_no(pair);
            }
            table_len = cuckoo_table.table_size[search_table_no];
            hash_val = ch::caculate_hash((uint32_t)key, search_table_no, table_len);
            bucket = cuckoo_table.table_group[search_table_no] + hash_val;
            flag = cg_search_in_bucket(key, values[group_index_in_all], bucket, group);
        }
    }

    DEVICEQUALIFIER INLINEQUALIFIER
        uint8_t
        cg_delete_in_bucket(const key_t& key, bucket_t* bucket, thread_block_tile<cg_size> group)
    {
        auto lane_id = group.thread_rank();
        uint32_t stride = bucket_size / cg_size;
        for (uint32_t ptr = 0; ptr < stride; ptr++) {
            key_t probe_data = (bucket->bucket_data)[ptr * cg_size + lane_id];
            key_t probe_key = probe_data;
            uint32_t group_mask = group.ballot(probe_key == key);
            if (group_mask != 0) {
                uint32_t group_leader = __ffs(group_mask) - 1;
                if (lane_id == group_leader) {
                    key_t empty_data = empty_key;
                    (bucket->bucket_data)[ptr * cg_size + lane_id] = empty_data;
                }
                return true;
            }
        }
        return false;
    }

    DEVICEQUALIFIER
    void cuckoo_delete(key_t* keys, uint32_t size)
    {
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);

        uint32_t pair, delete_table_no, table_len, hash_val;
        bucket_t* bucket;
        uint8_t flag = false;
        for (; group_index_in_all < size; group_index_in_all += step) {
            key_t key = keys[group_index_in_all];
            pair = ch::get_pair((uint32_t)key);
            if (key & 1) {
                delete_table_no = ch::get_table2_no(pair);
            } else {
                delete_table_no = ch::get_table1_no(pair);
            }
            table_len = cuckoo_table.table_size[delete_table_no];
            hash_val = ch::caculate_hash((uint32_t)key, delete_table_no, table_len);
            bucket = cuckoo_table.table_group[delete_table_no] + hash_val;
            flag = cg_delete_in_bucket(key, bucket, group);
            if (group.any(flag == true)) {
                continue;
            }
            if (key & 1) {
                delete_table_no = ch::get_table1_no(pair);
            } else {
                delete_table_no = ch::get_table2_no(pair);
            }
            table_len = cuckoo_table.table_size[delete_table_no];
            hash_val = ch::caculate_hash((uint32_t)key, delete_table_no, table_len);
            bucket = cuckoo_table.table_group[delete_table_no] + hash_val;
            flag = cg_delete_in_bucket(key, bucket, group);
        }
    }

    /**
     * new table has been set to cuckoo table; and old table has been replaced;
     * */
    DEVICEQUALIFIER
    void cuckoo_resize_up(bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
    {
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        auto lane_id = group.thread_rank();

        bucket_t* new_table = cuckoo_table.table_group[table_to_resize_no];
        uint32_t new_table_len = cuckoo_table.table_size[table_to_resize_no];
        for (; group_index_in_all < old_table_bucket_num; group_index_in_all += step) {
            bucket_t* bucket = old_table + group_index_in_all;
            for (int32_t ptr = lane_id; ptr < bucket_size; ptr += cg_size) {
                key_t data = (bucket->bucket_data)[ptr];
                key_t key = data;
                if (key != empty_key) {
                    uint32_t new_hash_val = ch::caculate_hash((uint32_t)key, table_to_resize_no, new_table_len);
                    // todo: maybe a bug
                    new_table[new_hash_val].bucket_data[ptr] = data;
                }
            }
        }
    }

    /**
     * resize down: copy first half of old table to new table
     * */
    DEVICEQUALIFIER
    void cuckoo_resize_down_pre(bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
    {
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        auto lane_id = group.thread_rank();
        bucket_t* new_table = cuckoo_table.table_group[table_to_resize_no];
        uint32_t new_table_bucket_num = cuckoo_table.table_size[table_to_resize_no];
        uint32_t ptr = group_index_in_all;
        bucket_t* old_bucket;
        bucket_t* new_bucket;
        for (; ptr < new_table_bucket_num; ptr += step) {
            old_bucket = old_table + ptr;
            new_bucket = new_table + ptr;
            for (int32_t cg_ptr = lane_id; cg_ptr < bucket_size; cg_ptr += cg_size) {
                (new_bucket->bucket_data)[cg_ptr] = (old_bucket->bucket_data)[cg_ptr];
            }
        }
    }

    /**
     * new table has been set to cuckoo table; and old table has been replaced;
     * */
    DEVICEQUALIFIER
    void cuckoo_resize_down(bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
    {
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        auto lane_id = group.thread_rank();

        uint32_t new_table_bucket_num = cuckoo_table.table_size[table_to_resize_no];
        uint32_t ptr;
        bucket_t* old_bucket;

        for (ptr = group_index_in_all + new_table_bucket_num; ptr < old_table_bucket_num; ptr += step) {
            old_bucket = old_table + ptr;
            for (int32_t cg_ptr = lane_id; cg_ptr < bucket_size; cg_ptr += cg_size) {
                uint8_t active = 0;
                key_t probe_data = (old_bucket->bucket_data)[cg_ptr];
                key_t probe_key = probe_data;
                if (probe_key != empty_key) {
                    active = 1;
                }
                auto group_mask = group.ballot(active == 1);
                while (group_mask != 0) {
                    auto leader = __ffs(group_mask) - 1;
                    key_t insert_data;
                    key_t insert_key;
                    insert_data = group.shfl(probe_data, leader);
                    insert_key = insert_data;
                    uint32_t insert_pre_table_no, pair;
                    pair = ch::get_pair((uint32_t)insert_key);
                    if (insert_key & 1) {
                        insert_pre_table_no = ch::get_table1_no(pair);
                    } else {
                        insert_pre_table_no = ch::get_table2_no(pair);
                    }
                    cg_inert(insert_data, insert_pre_table_no, group);
                    if (lane_id == leader)
                        active = 0;
                    group_mask = group.ballot(active == 1);
                }
            }
        }
    }

    void invoke_cuckoo_insert(uint32_t block_num, uint32_t thread_num, key_t* keys, uint32_t data_num);
    void invoke_cuckoo_search(uint32_t block_num, uint32_t thread_num, key_t* keys, uint32_t* values, uint32_t size);
    void invoke_cuckoo_delete(uint32_t block_num, uint32_t thread_num, key_t* keys, uint32_t size);
    void invoke_cuckoo_resize_up(uint32_t block_num, uint32_t thread_num, bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no);
    void invoke_cuckoo_resize_down_pre(uint32_t block_num, uint32_t thread_num, bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no);
    void invoke_cuckoo_resize_down(uint32_t block_num, uint32_t thread_num, bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no);

    HOSTQUALIFIER INLINEQUALIFIER void meta_data_to_device(cuckoo_t& host_ptr)
    {
        //cudaMemcpyToSymbol(cuckoo_table, &host_ptr, sizeof(cuckoo_t));
        //cudaDeviceSynchronize();
        //checkCudaErrors(cudaGetLastError());
        cuckoo_table = host_ptr;
    }

    HOSTQUALIFIER INLINEQUALIFIER void meta_data_to_device(error_table_t& host_ptr)
    {
        //cudaMemcpyToSymbol(error_table, &host_ptr, sizeof(error_table_t));
        //cudaDeviceSynchronize();
        //checkCudaErrors(cudaGetLastError());
        error_table = host_ptr;
    }
};

#include "DyCuckoo/core/dynamic_hash.cuh"
#include <cstdint>

template <typename T>
__global__ void kernel_cuckoo_insert(T dynamicHash, typename T::key_t* keys, uint32_t data_num)
{
    dynamicHash.cuckoo_insert(keys, data_num);
}

template <typename T>
__global__ void kernel_cuckoo_search(T dynamicHash, typename T::key_t* keys, uint32_t* values, uint32_t size)
{
    dynamicHash.cuckoo_search(keys, values, size);
}

template <typename T>
__global__ void kernel_cuckoo_delete(T dynamicHash, typename T::key_t* keys, uint32_t size)
{
    dynamicHash.cuckoo_delete(keys, size);
}

template <typename T>
__global__ void kernel_cuckoo_resize_up(T dynamicHash, typename T::bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
{
    dynamicHash.cuckoo_resize_up(old_table, old_table_bucket_num, table_to_resize_no);
}

template <typename T>
__global__ void kernel_cuckoo_resize_down_pre(T dynamicHash, typename T::bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
{
    dynamicHash.cuckoo_resize_down_pre(old_table, old_table_bucket_num, table_to_resize_no);
}

template <typename T>
__global__ void kernel_cuckoo_resize_down(T dynamicHash, typename T::bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
{
    dynamicHash.cuckoo_resize_down(old_table, old_table_bucket_num, table_to_resize_no);
}

template <typename MyDataLayout>
void DynamicHash<MyDataLayout>::invoke_cuckoo_insert(uint32_t block_num, uint32_t thread_num, key_t* keys, uint32_t data_num)
{
    kernel_cuckoo_insert<<<block_num, thread_num>>>(*this, keys, data_num);
}

template <typename MyDataLayout>
void DynamicHash<MyDataLayout>::invoke_cuckoo_search(uint32_t block_num, uint32_t thread_num, key_t* keys, uint32_t* values, uint32_t size)
{
    kernel_cuckoo_search<<<block_num, thread_num>>>(*this, keys, values, size);
}

template <typename MyDataLayout>
void DynamicHash<MyDataLayout>::invoke_cuckoo_delete(uint32_t block_num, uint32_t thread_num, key_t* keys, uint32_t size)
{
    kernel_cuckoo_delete<<<block_num, thread_num>>>(*this, keys, size);
}

template <typename MyDataLayout>
void DynamicHash<MyDataLayout>::invoke_cuckoo_resize_up(uint32_t block_num, uint32_t thread_num, bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
{
    kernel_cuckoo_resize_up<<<block_num, thread_num>>>(*this, old_table, old_table_bucket_num, table_to_resize_no);
}

template <typename MyDataLayout>
void DynamicHash<MyDataLayout>::invoke_cuckoo_resize_down_pre(uint32_t block_num, uint32_t thread_num, bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
{
    kernel_cuckoo_resize_down_pre<<<block_num, thread_num>>>(*this, old_table, old_table_bucket_num, table_to_resize_no);
}

template <typename MyDataLayout>
void DynamicHash<MyDataLayout>::invoke_cuckoo_resize_down(uint32_t block_num, uint32_t thread_num, bucket_t* old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no)
{
    kernel_cuckoo_resize_down<<<block_num, thread_num>>>(*this, old_table, old_table_bucket_num, table_to_resize_no);
}

#endif