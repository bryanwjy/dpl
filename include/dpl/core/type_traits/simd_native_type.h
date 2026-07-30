// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/simd_abi_traits.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct simd_native_type {};
template <typename T>
struct simd_native_type<T const> : simd_native_type<T> {};
template <typename T>
struct simd_native_type<T volatile> : simd_native_type<T> {};
template <typename T>
struct simd_native_type<T const volatile> : simd_native_type<T> {};
template <typename T>
struct simd_native_type<T&> : simd_native_type<T> {};
template <typename T>
struct simd_native_type<T&&> : simd_native_type<T> {};

template <typename T>
using simd_native_type_t = typename simd_native_type<T>::type;

template <typename T>
requires enable_simd_vector<T>
struct simd_native_type<T> {
    using type DPL_NODEBUG = typename simd_abi_traits<T>::native_vector;
};

template <typename T>
requires enable_simd_mask<T>
struct simd_native_type<T> {
    using type DPL_NODEBUG = typename simd_abi_traits<T>::native_mask;
};

template <typename T>
requires enable_simd_tuple<T>
struct simd_native_type<T> {
    using type DPL_NODEBUG = typename simd_abi_traits<T>::template native_tuple<
        std::tuple_size_v<T>>;
};

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
