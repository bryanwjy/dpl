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
        return __DPL forward<T>(expr).evaluate();
    }

    template <simd_expression T>
    requires (!member_evaluatable<T>) && unqualified_evaluatable<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr expression_result_t<T>
        DPL_VECTORCALL operator()(T&& expr) noexcept(
            noexcept(evaluate(__DPL forward<T>(expr)))) {
        return evaluate(__DPL forward<T>(expr));
    }

    template <typename T>
    requires simd_class<remove_cvref_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T&& val) noexcept {
        return __DPL forward<T>(expr);
    }
};

template <typename T>
concept member_evaluate_all = simd_expression<T> && requires(T&& expr) {
    {
        __DPL forward<T>(expr).evaluate_all()
    } -> same_as<expression_result_t<T>>;
};

template <typename T>
concept unqualified_evaluate_all = simd_expression<T> && requires(T&& expr) {
    {
        evaluate_all(__DPL forward<T>(expr))
    } -> same_as<expression_result_t<T>>;
};

struct evaluate_all_t {
private:
    template <simd_expression T>
    requires member_evaluate_all<expression_result_t<T>> ||
        unqualified_evaluate_all<expression_result_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto fallback(T&& expr) noexcept(
        is_nothrow_invocable_v<evaluate_all_t, expression_result_t<T>>) {
        return evaluate_all_t::operator()(
            dx::evaluate(__DPL forward<T>(expr)));
        if constexpr (member_evaluate_all<expression_result_t<T>> ||
            unqualified_evaluate_all<expression_result_t<T>>) {

        } else {
            return evaluate_all_t::fallback(
                dx::evaluate(__DPL forward<T>(expr)));
        }
    }

    template <simd_expression T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto fallback(T&& expr) {
        return evaluate_all_t::fallback(dx::evaluate(__DPL forward<T>(expr)));
    }

    template <typename T>
    requires (!simd_expression<T>) && simd_class<remove_cvref_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr decay_t<T> fallback(T&& simd) noexcept {
        return simd;
    }

public:
    template <simd_expression T>
    requires member_evaluate_all<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& expr) noexcept(
        noexcept(__DPL forward<T>(expr).evaluate_all())) {
        return __DPL forward<T>(expr).evaluate_all();
    }

    template <simd_expression T>
    requires (!member_evaluate_all<T>) && unqualified_evaluate_all<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& expr) noexcept(
        noexcept(evaluate_all(__DPL forward<T>(expr)))) {
        return evaluate_all(__DPL forward<T>(expr));
    }

    template <simd_expression T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& expr) {
        return evaluate_all_t::fallback(__DPL forward<T>(expr));
    }

    template <typename T>
    requires simd_class<remove_cvref_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T&& val) noexcept {
        return __DPL forward<T>(expr);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::evaluate_t evaluate{};
inline constexpr internal::evaluate_all_t evaluate_all{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
