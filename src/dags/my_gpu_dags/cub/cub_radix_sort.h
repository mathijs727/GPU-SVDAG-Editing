#pragma once
#include <gsl/gsl-lite.hpp>

template <typename T>
void cubDeviceRadixSortKeys(gsl::span<const T> inKeys, gsl::span<T> outKeys, cudaStream_t stream);