// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct simd_value_type {};
DPL_EXPORT template <typename T>
struct simd_value_type<T const> : simd_value_type<T> {};
DPL_EXPORT template <typename T>
struct simd_value_type<T volatile> : simd_value_type<T> {};
DPL_EXPORT template <typename T>
struct simd_value_type<T const volatile> : simd_value_type<T> {};
DPL_EXPORT template <typename T>
struct simd_value_type<T&> : simd_value_type<T> {};
DPL_EXPORT template <typename T>
struct simd_value_type<T&&> : simd_value_type<T> {};

DPL_EXPORT template <typename T>
using simd_value_type_t = typename simd_value_type<T>::type;

DPL_EXPORT template <typename T>
requires enable_simd_vector<T> || enable_simd_mask<T>
struct simd_value_type<T> {
    using type DPL_NODEBUG = typename T::value_type;
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
