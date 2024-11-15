#ifndef DATA_LAYOUT_H
#define DATA_LAYOUT_H
#include "DyCuckoo/dynamic_helpers.cuh"
#include "cnmem.h"
#include <limits>
#include <helper_cuda.h>

template<
        typename Key = uint32_t,
        Key EmptyKey = 0,
        uint32_t BucketSize = 16,
        uint8_t TableNum = 4,
        uint32_t errorTableLen = 10000
        >
class DataLayout{
public:
    using key_t = Key;

    static constexpr key_t empty_key = EmptyKey;

    static const uint32_t bucket_size = BucketSize;

    static const uint32_t error_table_len = errorTableLen;

    static const uint8_t table_num = TableNum;

public:

    class bucket_t{
    public:
        key_t bucket_data[bucket_size];
    };
public:
    class cuckoo_t{
    public:
        bucket_t* table_group[table_num];
        //count bucket num in single table
        uint32_t table_size[table_num];

        HOSTQUALIFIER
        static void device_table_mem_init(cuckoo_t &mycuckoo, uint32_t single_table_size){
            for(uint32_t i = 0; i < table_num; i++){
                cnmemMalloc((void**) &(mycuckoo.table_group[i]), sizeof(bucket_t) * single_table_size, 0);
                cudaMemset(mycuckoo.table_group[i], 0, sizeof(bucket_t) * single_table_size);
                mycuckoo.table_size[i] = single_table_size;
            }
            checkCudaErrors(cudaGetLastError());
        }

    };

public:
    class error_table_t{
    public:
        key_t * error_keys;
        //value_t* error_values;
        uint32_t error_pt;

        HOSTQUALIFIER
        void device_mem_init(){
            error_pt = 0;
            cnmemMalloc((void**)&error_keys, sizeof(key_t) * error_table_len, 0);
            //cnmemMalloc((void**)&error_values, sizeof(value_t)* error_table_len, 0);
            checkCudaErrors(cudaGetLastError());
        }

    };
};

#endif