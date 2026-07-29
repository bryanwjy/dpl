// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/simd_vector_type.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct simd_element_type {};
template <typename T>
struct simd_element_type<T const> : simd_element_type<T> {};
template <typename T>
struct simd_element_type<T volatile> : simd_element_type<T> {};
template <typename T>
struct simd_element_type<T const volatile> : simd_element_type<T> {};
template <typename T>
struct simd_element_type<T&> : simd_element_type<T> {};
template <typename T>
struct simd_element_type<T&&> : simd_element_type<T> {};

template <typename T>
using simd_element_type_t = typename simd_element_type<T>::type;

template <typename T>
requires enable_simd_vector<T>
struct simd_element_type<T> {
    using type DPL_NODEBUG = typename T::value_type;
};

template <typename T>
requires enable_simd_mask<T> && (!enable_simd_vector<T>)
struct simd_element_type<T> : simd_element_type<simd_vector_type_t<T>> {};

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
