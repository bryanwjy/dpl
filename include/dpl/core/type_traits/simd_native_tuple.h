// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/simd_abi_traits.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T, size_t N>
struct simd_native_tuple {};
template <typename T, size_t N>
struct simd_native_tuple<T const, N> : simd_native_tuple<T, N> {};
template <typename T, size_t N>
struct simd_native_tuple<T volatile, N> : simd_native_tuple<T, N> {};
template <typename T, size_t N>
struct simd_native_tuple<T const volatile, N> : simd_native_tuple<T, N> {};
template <typename T, size_t N>
struct simd_native_tuple<T&, N> : simd_native_tuple<T, N> {};
template <typename T, size_t N>
struct simd_native_tuple<T&&, N> : simd_native_tuple<T, N> {};

template <typename T, size_t N>
using simd_native_tuple_t = typename simd_native_tuple<T, N>::type;

template <typename T, size_t N>
requires enable_simd_vector<T>
struct simd_native_tuple<T, N> {
    using type DPL_NODEBUG =
        typename simd_abi_traits<T>::template native_tuple<N>;
};

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
