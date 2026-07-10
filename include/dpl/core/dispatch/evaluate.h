// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct evaluate_t {
    template <simd_expression T>
    requires member_evaluatable<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_expression_result_t<T>
        DPL_VECTORCALL operator()(T&& expr) noexcept(
            noexcept(__DPL forward<T>(expr).evaluate())) {
        return __DPL forward<T>(expr).evaluate();
    }

    template <simd_expression T>
    requires (!member_evaluatable<T>) && unqualified_evaluatable<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_expression_result_t<T>
        DPL_VECTORCALL operator()(T&& expr) noexcept(
            noexcept(evaluate(__DPL forward<T>(expr)))) {
        return evaluate(__DPL forward<T>(expr));
    }

    template <simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T&& val) noexcept {
        return __DPL forward<T>(val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::evaluate_t evaluate{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
