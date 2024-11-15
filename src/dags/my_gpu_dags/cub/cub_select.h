#pragma once
#include <gsl/gsl-lite.hpp>

template <typename T>
void cubDeviceSelectUnique(gsl::span<const T> inKeys, gsl::span<T> outKeys, uint32_t* pNumOutKeys, cudaStream_t stream);
