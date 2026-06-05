// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/is_canonical_type.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace internal {

template <typename T>
concept has_expression_result =
    (enable_simd_mask<T> || enable_simd_vector<T>) && !is_canonical_type_v<T> &&
    requires { typename T::result_type; } &&
    different_from<typename T::result_type, T> &&
    (enable_simd_mask<typename T::result_type> ||
        enable_simd_vector<typename T::result_type>);

template <typename T>
struct expression_result {};
template <typename T>
struct expression_result<T&> : expression_result<T> {};
template <typename T>
struct expression_result<T&&> : expression_result<T> {};
template <typename T>
struct expression_result<T const> : expression_result<T> {};
template <typename T>
struct expression_result<T volatile> : expression_result<T> {};
template <typename T>
struct expression_result<T const volatile> : expression_result<T> {};

template <has_expression_result T>
struct expression_result<T> {
    using type DPL_NODEBUG = typename T::result_type;
};

template <typename T>
using expression_result_t = typename expression_result<T>::type;

} // namespace internal

DPL_EXPORT template <typename T>
struct simd_expression_result : internal::expression_result<T> {};
DPL_EXPORT template <typename T>
using simd_expression_result_t = typename simd_expression_result<T>::type;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
