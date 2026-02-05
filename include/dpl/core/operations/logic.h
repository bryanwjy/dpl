// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/logic_reduction.h" // IWYU pragma: export
#include "dpl/core/operations/negated_mask.h"    // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_mask_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

struct logical_and_t {
    template <simd_mask_type L, simd_mask_type R>
    requires requires(L lhs, R rhs) { dx::bwand(lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        return dx::bwand(lhs, rhs);
    }
};

struct logical_or_t {
    template <simd_mask_type L, simd_mask_type R>
    requires requires(L lhs, R rhs) { dx::bwor(lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        return dx::bwor(lhs, rhs);
    }
};

struct logical_not_t {
    template <simd_mask_type M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M val) noexcept {
        return negated_mask<M>(+val);
    }

    template <simd_mask_type M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(negated_mask<M> val) noexcept {
        return !val;
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::logical_and_t logical_and{};
DPL_EXPORT inline constexpr internal::logical_or_t logical_or{};
DPL_EXPORT inline constexpr internal::logical_not_t logical_not{};
} // namespace cpo

DPL_EXPORT template <typename D>
class logical_simd_interface {
public:
    template <simd_mask_type R>
    requires regular_invocable<internal::logical_or_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_or_t, D, R> operator||(
        this D lhs, R rhs) noexcept
    requires simd_mask_type<D>
    {
        return datapar::logical_or(lhs, rhs);
    }

    template <simd_mask_type R>
    requires regular_invocable<internal::logical_and_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_and_t, D, R> operator&&(
        this D lhs, R rhs) noexcept
    requires simd_mask_type<D>
    {
        return datapar::logical_and(lhs, rhs);
    }

    template <simd_mask_type Self>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::logical_not_t, Self> operator!(
        this Self lhs) noexcept {
        return datapar::logical_not(lhs);
    }

    template <simd_mask_type L>
    requires regular_invocable<internal::logical_or_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::logical_or_t, L, D> operator||(
        L lhs, D rhs) noexcept
    requires simd_mask_type<D>
    {
        return datapar::logical_or(lhs, rhs);
    }

    template <simd_mask_type L>
    requires regular_invocable<internal::logical_or_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::logical_and_t, L, D> operator&&(
        L lhs, D rhs) noexcept
    requires simd_mask_type<D>
    {
        return datapar::logical_and(lhs, rhs);
    }
};

/**
 * The following are not exported by design
 */
template <simd_mask_type L, simd_mask_type R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwor_t, L, R> operator||(
    L lhs, R rhs) noexcept {
    return datapar::logical_or(lhs, rhs);
}

template <simd_mask_type L, simd_mask_type R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwand_t, L, R> operator&&(
    L lhs, R rhs) noexcept {
    return datapar::logical_and(lhs, rhs);
}

template <simd_mask_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto operator!(T val) noexcept
    -> invoke_result_t<internal::logical_not_t, T> {
    return datapar::logical_not(val);
}

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
