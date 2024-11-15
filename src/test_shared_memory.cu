#include "test_shared_memory.h"
#include <chrono>
#include <cuda.h>
#include <iostream>

__global__ void writeZero2Zero(float* p)
{
    p[0] = 0;
}

void testDeviceSyncPerformance(const char* name)
{
    using clock = std::chrono::high_resolution_clock;

    float* x;
    cudaMallocManaged(&x, 4llu);
    const auto start = clock::now();
    for (int i = 0; i < 100; i++) {
        dim3 block(1), grid(1);
        writeZero2Zero<<<grid, block>>>(x);
        cudaDeviceSynchronize();
    }
    cudaFree(x);

    const auto end = clock::now();
    std::chrono::duration<float> diff = end - start;
    std::cout << "[" << name << "] cudaDeviceSynchronize timing avg: " << (diff.count() * 10) << " ms\n";
}

__global__ void copyData(uint32_t* in, uint32_t* out)
{
    in[0] = out[0];
}

void testDeviceSyncPerformance2(const char* name, void* in)
{
    using clock = std::chrono::high_resolution_clock;

    uint32_t* out;
    cudaMallocManaged(&out, sizeof(uint32_t));
    const auto start = clock::now();
    for (int i = 0; i < 100; i++) {
        dim3 block(1), grid(1);
        copyData<<<grid, block>>>((uint32_t*)in, out);
        cudaDeviceSynchronize();
    }
    std::cout << "out: " << *out << std::endl;
    cudaFree(out);

    const auto end = clock::now();
    std::chrono::duration<float> diff = end - start;
    std::cout << "[" << name << "] cudaDeviceSynchronize timing avg: " << (diff.count() * 10) << " ms\n";
}
