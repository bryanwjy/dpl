// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/equivalence.h"
#include "dpl/core/concepts/extended.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_type.h"
#include "dpl/core/concepts/simd_vector.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/simd_expression_result.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

namespace internal {

void evaluate(...) noexcept = delete;

template <typename T>
concept has_expression_result_type = requires {
    typename simd_expression_result_t<T>;
} && simd_type<simd_expression_result_t<T>>;

template <typename T>
concept member_evaluatable = requires(T&& expr) {
    {
        __DPL forward<T>(expr).evaluate()
    } -> same_as<simd_expression_result_t<T>>;
};

template <typename T>
concept unqualified_evaluatable = requires(T&& expr) {
    { evaluate(__DPL forward<T>(expr)) } -> same_as<expression_result_t<T>>;
};

} // namespace internal

namespace atom {
template <typename T>
concept simd_expression = __DPL datapar::internal::has_expression_result<T> &&
    (__DPL datapar::internal::member_evaluatable<T> ||
        __DPL datapar::internal::unqualified_evaluatable<T>);
}

DPL_EXPORT template <typename T>
concept simd_expression = extended_simd_type<remove_cvref_t<T>> &&
    atom::simd_expression<remove_cvref_t<T>> &&
    !atom::simd_expression<simd_expression_result_t<T>> &&
    equivalent_simd_type_with<simd_expression_result_t<T>, remove_cvref_t<T>>;

DPL_EXPORT template <typename T>
concept mask_expression = simd_mask<remove_cvref_t<T>> && simd_expression<T>;

DPL_EXPORT template <typename T>
concept vector_expression =
    simd_vector<remove_cvref_t<T>> && simd_expression<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
