// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/details/type_traits.h"
#  include "dpl/core/type_interface/enable_simd_mask.h"
#  include "dpl/core/type_interface/enable_simd_vector.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct simd_expression_result {};
DPL_EXPORT template <typename T>
using simd_expression_result_t = typename simd_expression_result<T>::type;

DPL_EXPORT template <typename T>
struct simd_expression_result<T&> : simd_expression_result<T> {};
DPL_EXPORT template <typename T>
struct simd_expression_result<T&&> : simd_expression_result<T> {};
DPL_EXPORT template <typename T>
struct simd_expression_result<T const> : simd_expression_result<T> {};
DPL_EXPORT template <typename T>
struct simd_expression_result<T volatile> : simd_expression_result<T> {};
DPL_EXPORT template <typename T>
struct simd_expression_result<T const volatile> : simd_expression_result<T> {};
DPL_EXPORT template <internal::has_expression_result T>
struct simd_expression_result<T> {
    using type DPL_NODEBUG = typename T::result_type;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
