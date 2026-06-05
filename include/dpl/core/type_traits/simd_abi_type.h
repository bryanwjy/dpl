// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_abi.h"
#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct simd_abi_type {};
DPL_EXPORT template <typename T>
struct simd_abi_type<T const> : simd_abi_type<T> {};
DPL_EXPORT template <typename T>
struct simd_abi_type<T volatile> : simd_abi_type<T> {};
DPL_EXPORT template <typename T>
struct simd_abi_type<T const volatile> : simd_abi_type<T> {};

DPL_EXPORT template <typename T>
using simd_abi_type_t = typename simd_abi_type<T>::type;

DPL_EXPORT template <typename T>
requires (enable_simd_vector<T> || enable_simd_mask<T>) && (!enable_simd_abi<T>)
struct simd_abi_type<T> {
    using type DPL_NODEBUG = typename T::abi_type;
};

DPL_EXPORT template <typename T>
requires enable_simd_abi<T>
struct simd_abi_type<T> {
    using type DPL_NODEBUG = T;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
