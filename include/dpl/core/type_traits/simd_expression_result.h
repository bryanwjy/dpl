
// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_expression.h" // IWYU pragma: export
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
struct simd_expression_result {};

DPL_EXPORT template <simd_expression T>
struct simd_expression_result<T> {
    using type DPL_NODEBUG = internal::expression_result_t<T>;
};

DPL_EXPORT template <simd_class T>
requires (!simd_expression<T>)
struct simd_expression_result<T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <simd_class T>
requires (!simd_expression<T>)
struct simd_expression_result<T&> : expression_result<T> {};
DPL_EXPORT template <simd_class T>
requires (!simd_expression<T>)
struct simd_expression_result<T&&> : expression_result<T> {};
DPL_EXPORT template <simd_class T>
requires (!simd_expression<T>)
struct simd_expression_result<T const> : expression_result<T> {};
DPL_EXPORT template <simd_class T>
requires (!simd_expression<T>)
struct simd_expression_result<T volatile> : expression_result<T> {};
DPL_EXPORT template <simd_class T>
requires (!simd_expression<T>)
struct simd_expression_result<T const volatile> : expression_result<T> {};

DPL_EXPORT template <typename T>
using simd_expression_result_t = typename simd_expression_result<T>::type;

DPL_EXPORT template <typename T>
struct simd_expression_end : simd_expression_result<T> {};

DPL_EXPORT template <simd_expression T>
requires simd_expression<simd_expression_result_t<T>>
struct simd_expression_end :
    simd_expression_end<simd_expression_result_t<T>> {};

DPL_EXPORT template <typename T>
using simd_expression_end_t = typename simd_expression_end<T>::type;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
