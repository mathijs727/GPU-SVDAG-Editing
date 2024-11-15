#pragma once
#include <gsl/gsl-lite.hpp>

template <typename T>
void cubInclusiveSumMinusOne(gsl::span<const T> items, gsl::span<T> result, cudaStream_t stream);

template <typename T>
void cubExclusiveSum(gsl::span<const T> items, gsl::span<T> result, cudaStream_t stream);
