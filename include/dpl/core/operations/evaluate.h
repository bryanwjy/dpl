// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct evaluate_t {
    template <simd_expression T>
    requires member_evaluatable<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr expression_result_t<T>
        DPL_VECTORCALL operator()(T&& expr) noexcept(
            noexcept(__DPL forward<T>(expr).evaluate())) {
        static_assert(!simd_expression<expression_result_t<T>>,
            "Expression evaluation cannot produce another expression");
        return __DPL forward<T>(expr).evaluate();
    }

    template <simd_expression T>
    requires (!member_evaluatable<T>) && unqualified_evaluatable<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr expression_result_t<T>
        DPL_VECTORCALL operator()(T&& expr) noexcept(
            noexcept(evaluate(__DPL forward<T>(expr)))) {
        static_assert(!simd_expression<expression_result_t<T>>,
            "Expression evaluation cannot produce another expression");
        return evaluate(__DPL forward<T>(expr));
    }

    template <typename T>
    requires simd_class<remove_cvref_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T&& val) noexcept {
        return __DPL forward<T>(val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::evaluate_t evaluate{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
