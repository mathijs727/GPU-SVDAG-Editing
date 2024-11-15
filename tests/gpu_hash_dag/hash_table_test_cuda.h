#pragma once
#include <gsl/gsl-lite.hpp>
#include <cstdint>

template <typename T>
void markActiveElements_cuda(T& hashTable, gsl::span<const uint32_t> locations);

template <typename T>
void add_cuda(T& hashTable, gsl::span<const uint32_t> elements, gsl::span<uint32_t> outLocations);
template <typename T>
void addAsWarp_cuda(T& hashTable, gsl::span<const uint32_t> elements, gsl::span<uint32_t> outLocations);

template <typename T>
void find_cuda(T& hashTable, gsl::span<const uint32_t> elements, gsl::span<uint32_t> outLocations);
template <typename T>
void findAsWarp_cuda(T& hashTable, gsl::span<const uint32_t> elements, gsl::span<uint32_t> outLocations);
