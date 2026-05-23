// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

// IWYU pragma: begin_exports
#include "dpl/core/operations/logical/all_of.h"
#include "dpl/core/operations/logical/any_of.h"
#include "dpl/core/operations/logical/logical_and.h"
#include "dpl/core/operations/logical/logical_not.h"
#include "dpl/core/operations/logical/logical_or.h"
#include "dpl/core/operations/logical/negated_mask.h"
#include "dpl/core/operations/logical/none_of.h"
#include "dpl/core/operations/logical/some_of.h"
// IWYU pragma: end_exports

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_mask.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename D>
class logical_simd_interface {
public:
    template <simd_mask R>
    requires regular_invocable<internal::logical_or_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_or_t, D, R> operator||(
        this D lhs, R rhs) noexcept
    requires simd_mask<D>
    {
        return datapar::logical_or(lhs, rhs);
    }

    template <simd_mask R>
    requires regular_invocable<internal::logical_and_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_and_t, D, R> operator&&(
        this D lhs, R rhs) noexcept
    requires simd_mask<D>
    {
        return datapar::logical_and(lhs, rhs);
    }

    template <simd_mask Self>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_not_t, Self> operator!(
        this Self lhs) noexcept {
        return datapar::logical_not(lhs);
    }

    template <simd_mask L>
    requires regular_invocable<internal::logical_or_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::logical_or_t, L, D> operator||(
        L lhs, D rhs) noexcept
    requires simd_mask<D>
    {
        return datapar::logical_or(lhs, rhs);
    }

    template <simd_mask L>
    requires regular_invocable<internal::logical_or_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::logical_and_t, L, D> operator&&(
        L lhs, D rhs) noexcept
    requires simd_mask<D>
    {
        return datapar::logical_and(lhs, rhs);
    }
};

DPL_EXPORT template <simd_mask L, simd_mask R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwor_t, L, R> operator||(
    L lhs, R rhs) noexcept {
    return datapar::logical_or(lhs, rhs);
}

DPL_EXPORT template <simd_mask L, simd_mask R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwand_t, L, R> operator&&(
    L lhs, R rhs) noexcept {
    return datapar::logical_and(lhs, rhs);
}

DPL_EXPORT template <simd_mask T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto operator!(T val) noexcept
    -> invoke_result_t<internal::logical_not_t, T> {
    return datapar::logical_not(val);
}

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
