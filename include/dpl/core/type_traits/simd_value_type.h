// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct simd_value_type {};
template <typename T>
struct simd_value_type<T const> : simd_value_type<T> {};
template <typename T>
struct simd_value_type<T volatile> : simd_value_type<T> {};
template <typename T>
struct simd_value_type<T const volatile> : simd_value_type<T> {};
template <typename T>
struct simd_value_type<T&> : simd_value_type<T> {};
template <typename T>
struct simd_value_type<T&&> : simd_value_type<T> {};

template <typename T>
using simd_value_type_t = typename simd_value_type<T>::type;

template <typename T>
requires (enable_simd_vector<T> || enable_simd_mask<T>) &&
    requires { typename T::value_type; }
struct simd_value_type<T> {
    using type DPL_NODEBUG = typename T::value_type;
};

template <typename T>
requires enable_simd_tuple<T>
struct simd_value_type<T> : simd_value_type<simd_element_type_t<T>> {};
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
