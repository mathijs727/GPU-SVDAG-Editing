#ifndef STATIC_CUCKOO_H
#define STATIC_CUCKOO_H
#include "DyCuckoo/data/data_layout.cuh"
#include "DyCuckoo/tools/gputimer.h"
#include "cnmem.h"
#include "dynamic_hash.cuh"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <stdint.h>
using namespace cuckoo_helpers;
using namespace hashers;

namespace ch = cuckoo_helpers;
template <
    typename _key_t = uint32_t,
    uint32_t ThreadNum = 512,
    uint32_t BlockNum = 512>
class StaticCuckoo {
public:
    using MyDataLayout = DataLayout<_key_t>;
    using key_t = MyDataLayout::key_t;
    using bucket_t = MyDataLayout::bucket_t;
    using cuckoo_t = MyDataLayout::cuckoo_t;
    using error_table_t = MyDataLayout::error_table_t;
    static constexpr key_t empty_key = MyDataLayout::empty_key;
    static constexpr uint32_t bucket_size = MyDataLayout::bucket_size;
    static constexpr uint8_t table_num = MyDataLayout::table_num;

    static constexpr uint32_t thread_num = ThreadNum;
    static constexpr uint32_t block_num = BlockNum;

    DynamicHash<MyDataLayout> dynamicHash;

    cuckoo_t* host_cuckoo_table;
    error_table_t* host_error_table;

    cnmemDevice_t device;

    StaticCuckoo(uint32_t init_kv_num)
    {
        /// cnmem init
        memset(&device, 0, sizeof(device));
        device.size = (size_t)4 * 1024 * 1024 * 1024; /// more =(size_t) (0.95*props.totalGlobalMem);
        cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
        checkCudaErrors(cudaGetLastError());

        uint32_t s = init_kv_num / (table_num * bucket_size);
        // bucket num in a table
        s = ch::nextPrime(s);
        uint32_t s_bucket = (s & 1) ? s + 1 : s;
        host_cuckoo_table = (cuckoo_t*)malloc(sizeof(cuckoo_t));
        cuckoo_t::device_table_mem_init(*host_cuckoo_table, s_bucket);
        dynamicHash.meta_data_to_device(*host_cuckoo_table);

        checkCudaErrors(cudaGetLastError());

        // error table
        host_error_table = new error_table_t;
        host_error_table->device_mem_init();
        dynamicHash.meta_data_to_device(*host_error_table);
    }
    ~StaticCuckoo()
    {
        for (uint32_t i = 0; i < table_num; i++) {
            cnmemFree(host_cuckoo_table->table_group[i], 0);
        }
        free(host_cuckoo_table);

        cnmemFree(host_error_table->error_keys, 0);
        free(host_error_table);

        cnmemRelease();
    }

    void batch_insert(key_t* keys_d, uint32_t size)
    {
        dynamicHash.invoke_cuckoo_insert(block_num, thread_num, keys_d, size);
    }
    void batch_search(key_t* keys_d, uint32_t* values_d, uint32_t size)
    {
        dynamicHash.invoke_cuckoo_search(block_num, thread_num, keys_d, values_d, size);
    }

    void hash_insert(key_t* keys, uint32_t size)
    {

        key_t* dev_keys;
        cnmemMalloc((void**)&dev_keys, sizeof(key_t) * size, 0);
        cudaMemcpy(dev_keys, keys, sizeof(key_t) * size, cudaMemcpyHostToDevice);

        GpuTimer timer;
        timer.Start();
        dynamicHash.invoke_cuckoo_insert(block_num, thread_num, dev_keys, size);
        timer.Stop();
        double diff = timer.Elapsed() * 1000000;
        printf("<insert> %.2f\n", (double)(size) / diff);

        cnmemFree(dev_keys, 0);
        checkCudaErrors(cudaGetLastError());
    }

    void hash_search(key_t* keys, uint32_t* values, uint32_t size)
    {

        key_t* dev_keys;
        uint32_t* dev_values;
        cnmemMalloc((void**)&dev_keys, sizeof(key_t) * size, 0);
        cudaMemcpy(dev_keys, keys, sizeof(key_t) * size, cudaMemcpyHostToDevice);
        cnmemMalloc((void**)&dev_values, sizeof(uint32_t) * size, 0);
        cudaMemset(dev_values, 0, sizeof(uint32_t) * size);

        GpuTimer timer;
        timer.Start();
        dynamicHash.invoke_cuckoo_search(block_num, thread_num, dev_keys, dev_values, size);
        timer.Stop();
        double diff = timer.Elapsed() * 1000000;
        printf("<search> %.2f\n", (double)(size) / diff);

        cudaMemcpy(values, dev_values, sizeof(uint32_t) * size, cudaMemcpyDeviceToHost);
        cnmemFree(dev_keys, 0);
        cnmemFree(dev_values, 0);
        checkCudaErrors(cudaGetLastError());
    }

    void hash_delete(key_t* keys, uint32_t size)
    {
        key_t* dev_keys;
        cnmemMalloc((void**)&dev_keys, sizeof(key_t) * size, 0);
        cudaMemcpy(dev_keys, keys, sizeof(key_t) * size, cudaMemcpyHostToDevice);

        dynamicHash.invoke_cuckoo_delete(block_num, thread_num, dev_keys, size);
        cnmemFree(dev_keys, 0);
        checkCudaErrors(cudaGetLastError());
    }
};

#endif