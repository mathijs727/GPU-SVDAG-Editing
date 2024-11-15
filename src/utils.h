#pragma once
#include "my_units.h"
#include "typedefs.h"
#include <chrono>
#include <functional>
#include <gsl/gsl-lite.hpp>
#include <random>
#include <type_traits>

#ifndef __CUDA_ARCH__
#include <immintrin.h>
#endif

namespace Utils {

template <uint32_t Offset, uint32_t Size>
HOST_DEVICE uint32_t extract_bits(uint32_t mask)
{
    static_assert(Offset + Size <= 32);

    if constexpr (Size == 0) {
        return 0;
    } else if constexpr (Offset == 0 && Size == 32) {
        return mask;
    } else {
        static constexpr uint32_t BitMask = (1u << Size) - 1u;
        return (mask >> Offset) & BitMask;
    }
}
template <uint32_t Offset, uint32_t Size>
HOST_DEVICE void insert_bits(uint32_t& mask, uint32_t bits)
{
    static_assert(Offset + Size <= 32);

    if constexpr (Size > 0) {
        if constexpr (Size < 32) {
            check(bits < (1u << Size));
        }

        mask |= (bits << Offset);
    }
}

// https://www.pcg-random.org/download.html
struct PCG_RNG {
public:
    HOST_DEVICE PCG_RNG(uint64_t stream)
        : inc(stream)
    {
        sampleU32();
        sampleU32();
    }

    HOST_DEVICE uint32_t sampleU32()
    {
        const uint64_t oldstate = this->state;
        // Advance internal state
        this->state = oldstate * 6364136223846793005ULL + (this->inc | 1);
        // Calculate output function (XSH RR), uses old state for max ILP
        const uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        const uint32_t rot = (uint32_t)(oldstate >> 59);
        // Replace "-rot" by "0xFFFFFFFF - rot + 1" to prevent compiler warnings:
        // https://stackoverflow.com/questions/8026694/c-unary-minus-operator-behavior-with-unsigned-operands
        return (xorshifted >> rot) | (xorshifted << ((0xFFFFFFFF - rot + 1) & 31));
    }

    HOST_DEVICE float sampleFloat()
    {
        return float(sampleU32()) / float(std::numeric_limits<uint32_t>::max());
    }
    HOST_DEVICE float2 sampleFloat2()
    {
        return make_float2(sampleFloat(), sampleFloat());
    }
    HOST_DEVICE float3 sampleFloat3()
    {
        return make_float3(sampleFloat(), sampleFloat(), sampleFloat());
    }

private:
    uint64_t state = 0x853c49e6748fea9bULL, inc;
};

// COPIED FROM: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
HOST_DEVICE unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// COPIED FROM: https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
HOST_DEVICE uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
    unsigned int xx = Utils::expandBits(x);
    unsigned int yy = Utils::expandBits(y);
    unsigned int zz = Utils::expandBits(z);
    return (xx << 2) | (yy << 1) | zz;
}

HOST_DEVICE uint64_t morton3D_64(uint32_t x, uint32_t y, uint32_t z)
{
    unsigned int xx = Utils::expandBits(x >> 10);
    unsigned int yy = Utils::expandBits(y >> 11);
    unsigned int zz = Utils::expandBits(z >> 11);
    const uint64_t first = morton3D(x, y, z);
    const uint64_t second = xx | (zz << 1) | (yy << 2);
    return first | (second << 32);
}
// CUDA compatible (std::)lower_bound which returns an index instead of an iterator.
template <typename T>
HOST_DEVICE uint32_t lower_bound(gsl::span<const T> elements, const T& searchElement)
{
    // Equivalent of std::lower_bound
    // https://en.cppreference.com/w/cpp/algorithm/lower_bound
    uint32_t count = (uint32_t)elements.size(), first = 0;
    while (count > 0) {
        uint32_t it = first;
        const uint32_t step = count / 2;
        it += step;
        if (elements[it] < searchElement) {
            first = ++it;
            count -= step + 1;
        } else
            count = step;
    }
    return first;
}

// CUDA compatible (std::)upper_bound which returns an index instead of an iterator.
// https://en.cppreference.com/w/cpp/algorithm/upper_bound
template <typename T>
HOST_DEVICE uint32_t upper_bound(gsl::span<const T> elements, const T& searchElement)
{
    uint32_t count = (uint32_t)elements.size(), first = 0;
    while (count > 0) {
        uint32_t it = first;
        const uint32_t step = count / 2;
        it += step;
        if (!(searchElement < elements[it])) {
            first = ++it;
            count -= step + 1;
        } else
            count = step;
    }
    return first;
}

HOST_DEVICE uint32_t divideRoundUp(uint32_t lhs, uint32_t rhs)
{
    return lhs == 0u ? 0u : ((lhs - 1u) / rhs) + 1u;
}
HOST_DEVICE uint64_t divideRoundUp(uint64_t lhs, uint64_t rhs)
{
    return lhs == 0llu ? 0llu : ((lhs - 1llu) / rhs) + 1llu;
}

HOST_DEVICE bool compare_u32_array(const uint32_t* lhs, const uint32_t* rhs, uint32_t sizeInU32)
{
    for (uint32_t i = 0; i < sizeInU32; ++i) {
        if (lhs[i] != rhs[i])
            return false;
    }
    return true;
}

#ifdef __CUDACC__
DEVICE bool compare_u32_array_warp(const uint32_t* lhs, const uint32_t* rhs, uint32_t sizeInU32, uint32_t threadRank)
{
    check(__activemask() == 0xFFFFFFFF);
    const uint32_t threadMask = __ballot_sync(0xFFFFFFFF, lhs[threadRank] == rhs[threadRank]);
    const uint32_t validMask = (1u << sizeInU32) - 1u;
    return (threadMask & validMask) == validMask;
}

DEVICE bool compare_u32_array_varying_warp(const uint32_t* pFixed, const uint32_t* pVarying, uint32_t sizeInU32, uint32_t threadRank, uint32_t activeMask)
{
#if 1
    // For some reason this is **much** faster than the warp comparison in real-world scenarios.
    bool out = true;
    if (activeMask & (1u << threadRank)) {
        for (uint32_t i = 0; i < sizeInU32; ++i) {
            out &= pFixed[i] == pVarying[i];
            if (!out)
                break;
        }
    }
    return out;
#else
    check(__activemask() == 0xFFFFFFFF);
    uint32_t myCompareMask = 0;
    const uint32_t numActive = __popc(activeMask);
    for (uint32_t i = 1; i <= numActive; ++i) {
        const uint32_t dstLane = __fns(activeMask, 0, i);
        const uint32_t* pCompare = std::bit_cast<const uint32_t*>(__shfl_sync(0xFFFFFFFF, std::bit_cast<uintptr_t>(pVarying), dstLane));
        const bool comp = threadRank < sizeInU32 ? pFixed[threadRank] == pCompare[threadRank] : false;
        // const bool comp = (threadRank < sizeInU32) * (pFixed[threadRank] == pCompare[threadRank]);
        const uint32_t compareMask = __ballot_sync(0xFFFFFFFF, comp);

        if (threadRank == dstLane)
            myCompareMask = compareMask;
    }
    const uint32_t validMask = (1u << sizeInU32) - 1u;
    return (myCompareMask & validMask) == validMask;
#endif
}
#endif

// CUDA compatible version of std::equal.
// Code copied from:
// https://en.cppreference.com/w/cpp/algorithm/equal
template <class InputIt1, class InputIt2>
HOST_DEVICE bool equal(InputIt1 first1, InputIt1 last1, InputIt2 first2)
{
    for (; first1 != last1; ++first1, ++first2) {
        if (!(*first1 == *first2)) {
            return false;
        }
    }
    return true;
}

HOST_DEVICE uint32_t rotate_left(uint32_t a, uint32_t amount)
{
    // Seems like it is not possible to tell CUDA to use C++17 for device code and >= C++20 for host code.
    // So no std::rotl for me :(
#if defined(__CUDA_ARCH__)
    // https://github.com/cbuchner1/CudaMiner/blob/master/salsa_kernel.cu
    return (a << amount) | (a >> (32 - amount));
#elif defined(__clang__)
    return __builtin_rotateleft32(a, amount);
#else
    return _rotl(a, amount);
#endif
}

HOST_DEVICE uint32_t rotate_right(uint32_t a, uint32_t amount)
{
    // Seems like it is not possible to tell CUDA to use C++17 for device code and >= C++20 for host code.
    // So no std::rotr for me :(
#if defined(__CUDA_ARCH__)
    // https://github.com/cbuchner1/CudaMiner/blob/master/salsa_kernel.cu
    return (a >> amount) | (a << (32 - amount));
#elif defined(__clang__)
    return __builtin_rotateright32(a, amount);
#else
    return _rotr(a, amount);
#endif
}

HOST_DEVICE uint32_t bit_width(uint32_t a)
{
    // Seems like it is not possible to tell CUDA to use C++17 for device code and >= C++20 for host code.
    // So no std::bit_width for me :(
#if defined(__CUDA_ARCH__)
    // Equivalent to std::bit_width
    return 32 - __clz(a);
#else
    uint32_t out = 0;
    while (a != 0) {
        a >>= 1;
        ++out;
    }
    return out;
#endif
}

static constexpr int constexprPopc(uint32_t i)
{
    int out = 0;
    for (int j = 0; j < 32; ++j) {
        if (i & (1u << j))
            ++out;
    }
    return out;
}

HOST_DEVICE uint32 popc(uint32 a)
{
#if defined(__CUDA_ARCH__)
    return __popc(a);
#else
#if USE_POPC_INTRINSICS
    return (uint32_t)__builtin_popcount(a);
#else // !USE_POPC_INTRINSICS
    // Source: http://graphics.stanford.edu/~seander/bithacks.html
    a = a - ((a >> 1) & 0x55555555);
    a = (a & 0x33333333) + ((a >> 2) & 0x33333333);
    return ((a + (a >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
#endif // ~ USE_POPC_INTRINSICS
#endif
}
HOST_DEVICE uint32 popcll(uint64 a)
{
#if defined(__CUDA_ARCH__)
    return __popcll(a);
#else
#if USE_POPC_INTRINSICS
    return uint32(__builtin_popcountl(a));
#else // !USE_POPC_INTRINSICS
    return popc(uint32(a >> 32)) + popc(uint32(a & 0xFFFFFFFF));
#endif // ~ USE_POPC_INTRINSICS
#endif
}
template <uint32 bit = 31>
HOST_DEVICE bool has_flag(uint32 index)
{
    return index & (1u << bit);
}
template <uint32 bit = 31>
HOST_DEVICE uint32 set_flag(uint32 index)
{
    return index | (1u << bit);
}
template <uint32 bit = 31>
HOST_DEVICE uint32 clear_flag(uint32 index)
{
    return index & ~(1u << bit);
}
HOST_DEVICE uint32 level_max_size(uint32 level)
{
    return 1u << (3 * (12 - level));
}
HOST_DEVICE uint8 child_mask(uint32 node)
{
    return node & 0xff;
}
HOST_DEVICE uint32 total_size(uint32 node)
{
    return Utils::popc(Utils::child_mask(node)) + 1;
}
HOST_DEVICE uint32 child_offset(uint8 childMask, uint8 child)
{
    return popc(childMask & ((1u << child) - 1u)) + 1;
}
HOST_DEVICE uint32 murmurhash32(uint32 h)
{
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
HOST_DEVICE uint64 murmurhash64(uint64 h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    return h;
}

#if USE_ALTERNATE_HASH
template <typename T>
HOST_DEVICE uint32 murmurhash32xN(const T ph, std::size_t n, uint32 seed = 0)
{
    uint32 h = seed;
    for (std::size_t i = 0; i < n; ++i) {
        uint32 k = ph[i];
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
    }

    h ^= uint32(n);
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
#endif // ~ USE_ALTERNATE_HASH

constexpr static uint32_t hash_x = 2717583853;
constexpr static uint32_t hash_y = 1117415409;
constexpr static uint32_t prime_divisor = 4294967291u;

HOST_DEVICE uint32_t slabHash(uint32_t item)
{
    return ((hash_x * item) + hash_y) % prime_divisor;
}
template <typename T>
HOST_DEVICE uint32_t xorHash32xN(T pItem, uint32_t itemSizeInU32)
{
    uint32_t seed = 0;
    for (uint32_t i = 0; i < itemSizeInU32; ++i) {
        const uint32_t partialHash = ((hash_x * pItem[i]) + i * hash_y) % prime_divisor;
        seed ^= partialHash;
    }
    return seed;
}
template <typename T>
HOST_DEVICE uint32_t xorMurmurHash32xN(T pItem, uint32_t itemSizeInU32)
{
    uint32_t seed = 0;
    for (uint32_t i = 0; i < itemSizeInU32; ++i)
        seed ^= Utils::murmurhash32(pItem[i] + i);
    return seed;
}
template <typename T>
HOST_DEVICE void hash_combine(size_t& seed, const T& v)
{
    // https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
template <typename T>
HOST void hash_combine_cpu(size_t& seed, const T& v)
{
    // https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
template <typename T>
HOST_DEVICE uint32_t boostCombineHash32xN(T pItem, uint32_t itemSizeInU32)
{
    uint32_t seed = 0;
    for (uint32_t i = 0; i < itemSizeInU32; ++i) {
        // Same hash as SlabHash
        const uint32_t partialHash = ((hash_x * pItem[i]) + hash_y) % prime_divisor;
        seed ^= partialHash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
#ifdef __CUDACC__
DEVICE uint32_t xorHash32xN_warp(uint32_t item, uint32_t itemSizeInU32, uint32_t threadRank)
{
    // Similar to SlabHash but make each element dependent on the threadRank/index.
    // This ensures that reordering the elements in the item gives different hash codes.
    uint32_t partialHash = ((hash_x * item) + threadRank * hash_y) % prime_divisor;
    if (threadRank >= itemSizeInU32)
        partialHash = 0;
    return __reduce_xor_sync(0xFFFFFFFF, partialHash);
}
DEVICE uint32_t xorMurmurHash32xN_warp(uint32_t item, uint32_t itemSizeInU32, uint32_t threadRank)
{
    uint32_t partialHash = Utils::murmurhash32(item + threadRank);
    if (threadRank >= itemSizeInU32)
        partialHash = 0;
    return __reduce_xor_sync(0xFFFFFFFF, partialHash);
}
DEVICE uint32_t boostCombineHash32xN_warp(uint32_t item, uint32_t itemSizeInU32, uint32_t threadRank)
{
    uint32_t partialHash = ((hash_x * item) + hash_y) % prime_divisor;
    uint32_t seed = 0;
    for (uint32_t i = 0; i < itemSizeInU32; ++i) {
        // Boost hash_combine
        seed ^= __shfl_sync(0xFFFFFFFF, partialHash, i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}
#endif

#if USE_BLOOM_FILTER
// See https://gist.github.com/badboy/6267743
HOST_DEVICE
uint64 hash64shift(uint64 key)
{
    key = (~key) + (key << 21);
    key = key ^ (key >> 24);
    // key = (key + (key << 3)) + (key << 8); // key * 265
    key *= 265;
    key = key ^ (key >> 14);
    // key = (key + (key << 2)) + (key << 4); // key * 21
    key *= 21;
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

// See https://github.com/ZilongTan/fast-hash/blob/master/fasthash.c
HOST_DEVICE
uint64 fasthash64_mix(uint64 aValue)
{
    aValue ^= aValue >> 23;
    aValue *= 0x2127599bf4325c37ull;
    aValue ^= aValue >> 47;
    return aValue;
}

HOST_DEVICE
uint64 fasthash64(uint32 const* buf, size_t len, uint64 seed = 0)
{
    uint64 const m = 0x880355f21e6d1965ULL;
    uint32 const* end = buf + (len & ~size_t(1));

    uint64 h = seed ^ (len * sizeof(uint32) * m);

    while (buf != end) {
        uint64 v;
        std::memcpy(&v, buf, sizeof(uint64));
        buf += 2;

        h ^= fasthash64_mix(v);
        h *= m;
    }

    if (len & 1) {
        uint64 v = 0;
        v ^= *buf;
        h ^= fasthash64_mix(v);
        h *= m;
    }

    return fasthash64_mix(h);
}

// See http://blog.michaelschmatz.com/2016/04/11/how-to-write-a-bloom-filter-cpp/
HOST_DEVICE
uint32 nth_hash(uint32 aN, uint32 aHashA, uint32 aHashB, uint32 aSize)
{
    return (aHashA + aN * aHashB) % aSize;
}
#endif // ~  USE_BLOOM_FILTER

template <typename T>
HOST_DEVICE size_t vector_size_bytes(const std::vector<T>& vector)
{
    return vector.size() * sizeof(T);
}

template <typename T>
HOST my_units::bytes vector_memory(const std::vector<T>& vector)
{
    return my_units::bytes(vector.size() * sizeof(T));
}

template <typename T>
HOST_DEVICE double to_MB(T bytes)
{
    return double(bytes) / double(1 << 20);
}
template <>
HOST_DEVICE double to_MB(my_units::bytes bytes)
{
    // return bytes.convert<units::data::megabyte>().value();
    return (double)bytes / 1000000;
}

HOST_DEVICE double to_KB(my_units::bytes bytes)
{
    // return bytes.convert<units::data::megabyte>().value();
    return (double)bytes / 1000;
}

HOST_DEVICE double to_B(my_units::bytes bytes)
{
    // return static_cast<double>(bytes.value());
    return (double)bytes;
}

HOST_DEVICE uint32 subtract_mod(uint32 value, uint32 max /* exclusive */)
{
    if (value == 0) {
        return max - 1;
    } else {
        return value - 1;
    }
}

template <typename T>
HOST_DEVICE constexpr T divide_ceil(T Dividend, T Divisor)
{
    return 1 + (Dividend - 1) / Divisor;
}

template <typename T>
HOST_DEVICE constexpr uint32_t sizeof_u32()
{
    static_assert(sizeof(T) % sizeof(uint32_t) == 0);
    return sizeof(T) / sizeof(uint32_t);
}

// Prefetching
using PfEphemeral = std::integral_constant<int, 0>;
using PfAllLevels = std::integral_constant<int, 3>;
using PfLowLocality = std::integral_constant<int, 1>;
using PfMedLocality = std::integral_constant<int, 2>;

template <class tLocality = PfAllLevels>
HOST void prefetch_ro(void const* aAddr, tLocality = tLocality {})
{
#if defined(__GNUC__)
    __builtin_prefetch(aAddr, 0, tLocality::value);
#else
    checkAlways(false);
#endif // ~ GCC
}

HOST double seconds()
{
    static auto start = std::chrono::high_resolution_clock::now();
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count()) / 1.e9;
}

HOST double to_ms(const std::chrono::duration<double>& duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

static constexpr uint32_t constexpr_log2(uint32_t v)
{
    for (uint32_t i = 0; i < 32; ++i) {
        if ((v >> i) & 1)
            return i;
    }
    return 0xFFFFFFFF;
}

// https://stackoverflow.com/questions/28997271/c11-way-to-index-tuple-at-runtime-without-using-switch
// Calls your func with tuple element.
template <class Func, class Tuple, size_t N = 0>
auto runtime_get(Func func, Tuple& tup, size_t idx)
{
    if (N == idx) {
        return std::invoke(func, std::get<N>(tup));
    }

    if constexpr (N + 1 < std::tuple_size_v<Tuple>) {
        return runtime_get<Func, Tuple, N + 1>(func, tup, idx);
    }
}

// https://stackoverflow.com/questions/48913092/constexpr-in-for-statement
template <typename T, T value>
struct ValueForward {
    static constexpr T v = value;
};
template <int... Is, typename F>
void constexpr_for_loop_impl(std::integer_sequence<int, Is...>, F&& f)
{
    // C++ 17 folding expression
    ((f(ValueForward<int, Is>())), ...);
}
template <int end, typename F>
void constexpr_for_loop(F&& f)
{
    constexpr_for_loop_impl(std::make_integer_sequence<int, end>(), std::forward<F>(f));
}

// https://www.modernescpp.com/index.php/visiting-a-std-variant-with-the-overload-pattern
template <typename... Ts>
struct visitor : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
visitor(Ts...) -> visitor<Ts...>;

template <typename T>
struct TypeForward {
};

}
