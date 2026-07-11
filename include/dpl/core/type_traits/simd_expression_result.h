// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/has_expression_result.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct simd_expression_result {};
template <typename T>
using simd_expression_result_t = typename simd_expression_result<T>::type;

template <typename T>
struct simd_expression_result<T&> : simd_expression_result<T> {};
template <typename T>
struct simd_expression_result<T&&> : simd_expression_result<T> {};
template <typename T>
struct simd_expression_result<T const> : simd_expression_result<T> {};
template <typename T>
struct simd_expression_result<T volatile> : simd_expression_result<T> {};
template <typename T>
struct simd_expression_result<T const volatile> : simd_expression_result<T> {};
template <internal::has_expression_result T>
struct simd_expression_result<T> {
    using type DPL_NODEBUG = typename T::result_type;
};

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
