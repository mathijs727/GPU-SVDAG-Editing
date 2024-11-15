#pragma once
#include <gsl/gsl-lite.hpp>

template <typename T1, typename T2>
void cubDeviceMergeSortPairs(gsl::span<T1> keys, gsl::span<T2> items, cudaStream_t stream);
