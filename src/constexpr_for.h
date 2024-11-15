#pragma once
#include <cstdint>
#include <utility>

template <auto V>
struct ValueForward {
    constexpr static auto value = V;
};

template <int... Is, typename F>
void constexpr_for_impl(std::integer_sequence<int, Is...>, F&& f)
{
    (f(ValueForward<Is>()), ...);
}

template <int N, typename F>
void constexpr_for(F&& f)
{
    constexpr_for_impl(std::make_integer_sequence<int, N>(), f);
}
