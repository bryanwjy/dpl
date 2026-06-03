// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_class.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_vector.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

namespace internal {

template <typename T>
concept has_expression_result =
    simd_class<T> && requires { typename T::result_type; } &&
    different_from<typename T::result_type, T> &&
    simd_class<typename T::result_type>;

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
    using type = typename T::result_type;
};

template <typename T>
using expression_result_t = typename expression_result<T>::type;

void evaluate(...) noexcept = delete;

template <typename T>
concept member_evaluatable =
    has_expression_result<remove_cvref_t<T>> && requires(T&& expr) {
        {
            __DPL forward<T>(expr).evaluate()
        } -> same_as<expression_result_t<T>>;
    };

template <typename T>
concept unqualified_evaluatable =
    has_expression_result<remove_cvref_t<T>> && requires(T&& expr) {
        {
            evaluate(__DPL forward<T>(expr))
        } -> same_as<expression_result_t<T>>;
    };

} // namespace internal

namespace atom {
template <typename T>
concept simd_expression = __DPL datapar::internal::has_expression_result<T> &&
    (__DPL datapar::internal::member_evaluatable<T> ||
        __DPL datapar::internal::unqualified_evaluatable<T>);
}

DPL_EXPORT template <typename T>
concept simd_expression = extended_class<T> && atom::simd_expression<T>;
DPL_EXPORT template <typename T>
concept mask_expression = simd_mask<T> && simd_expression<T>;
DPL_EXPORT template <typename T>
concept vector_expression = simd_vector<T> && simd_expression<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
