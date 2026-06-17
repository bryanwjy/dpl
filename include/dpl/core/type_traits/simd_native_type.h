// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/simd_abi_traits.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct simd_native_type {};
DPL_EXPORT template <typename T>
struct simd_native_type<T const> : simd_native_type<T> {};
DPL_EXPORT template <typename T>
struct simd_native_type<T volatile> : simd_native_type<T> {};
DPL_EXPORT template <typename T>
struct simd_native_type<T const volatile> : simd_native_type<T> {};
DPL_EXPORT template <typename T>
struct simd_native_type<T&> : simd_native_type<T> {};
DPL_EXPORT template <typename T>
struct simd_native_type<T&&> : simd_native_type<T> {};

DPL_EXPORT template <typename T>
using simd_native_type_t = typename simd_native_type<T>::type;

DPL_EXPORT template <typename T>
requires enable_simd_vector<T>
struct simd_native_type<T> {
    using type DPL_NODEBUG = typename simd_abi_traits<T>::native_vector;
};

DPL_EXPORT template <typename T>
requires enable_simd_mask<T>
struct simd_native_type<T> {
    using type DPL_NODEBUG = typename simd_abi_traits<T>::native_mask;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
