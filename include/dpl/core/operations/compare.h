// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/type_traits/details/cpo_result.h"
#endif

// IWYU pragma: begin_exports
#include "dpl/core/operations/compare/cmpeq.h"
#include "dpl/core/operations/compare/cmpge.h"
#include "dpl/core/operations/compare/cmpgt.h"
#include "dpl/core/operations/compare/cmple.h"
#include "dpl/core/operations/compare/cmplt.h"
#include "dpl/core/operations/compare/cmpneq.h"
#include "dpl/core/operations/compare/cmpunord.h"
// IWYU pragma: end_exports

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

template <typename D>
class comparison_simd_interface {
public:
    template <typename R>
    requires internal::cpo_invocable<internal::cmpeq_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr internal::cpo_result_t<internal::cmpeq_t, D, R> operator==(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpeq(lhs, rhs);
    }

    template <typename R>
    requires internal::cpo_invocable<internal::cmpneq_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr internal::cpo_result_t<internal::cmpneq_t, D, R> operator!=(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpneq(lhs, rhs);
    }

    template <typename R>
    requires internal::cpo_invocable<internal::cmplt_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr internal::cpo_result_t<internal::cmplt_t, D, R> operator<(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmplt(lhs, rhs);
    }

    template <typename R>
    requires internal::cpo_invocable<internal::cmple_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr internal::cpo_result_t<internal::cmple_t, D, R> operator<=(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmple(lhs, rhs);
    }

    template <typename R>
    requires internal::cpo_invocable<internal::cmpgt_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr internal::cpo_result_t<internal::cmpgt_t, D, R> operator>(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpgt(lhs, rhs);
    }

    template <typename R>
    requires internal::cpo_invocable<internal::cmpge_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr internal::cpo_result_t<internal::cmpge_t, D, R> operator>=(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpge(lhs, rhs);
    }

    template <typename L>
    requires internal::cpo_invocable<internal::cmpeq_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr internal::cpo_result_t<internal::cmpeq_t, L, D> operator==(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpeq(lhs, rhs);
    }

    template <typename L>
    requires internal::cpo_invocable<internal::cmpneq_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr internal::cpo_result_t<internal::cmpneq_t, L, D>
    operator!=(L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpneq(lhs, rhs);
    }

    template <typename L>
    requires internal::cpo_invocable<internal::cmplt_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr internal::cpo_result_t<internal::cmplt_t, L, D> operator<(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmplt(lhs, rhs);
    }

    template <typename L>
    requires internal::cpo_invocable<internal::cmple_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr internal::cpo_result_t<internal::cmple_t, L, D> operator<=(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmple(lhs, rhs);
    }

    template <typename L>
    requires internal::cpo_invocable<internal::cmpgt_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr internal::cpo_result_t<internal::cmpgt_t, L, D> operator>(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpgt(lhs, rhs);
    }

    template <typename L>
    requires internal::cpo_invocable<internal::cmpge_t, L, D>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr internal::cpo_result_t<internal::cmpge_t, L, D> operator>=(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::cmpge(lhs, rhs);
    }
};

inline namespace operators {
template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr internal::cpo_result_t<internal::cmpeq_t, L, R> operator==(
    L lhs, R rhs) noexcept {
    return datapar::cmpeq(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr internal::cpo_result_t<internal::cmpneq_t, L, R> operator!=(
    L lhs, R rhs) noexcept {
    return datapar::cmpneq(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr internal::cpo_result_t<internal::cmplt_t, L, R> operator<(
    L lhs, R rhs) noexcept {
    return datapar::cmplt(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr internal::cpo_result_t<internal::cmple_t, L, R> operator<=(
    L lhs, R rhs) noexcept {
    return datapar::cmple(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr internal::cpo_result_t<internal::cmpgt_t, L, R> operator>(
    L lhs, R rhs) noexcept {
    return datapar::cmpgt(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr internal::cpo_result_t<internal::cmpge_t, L, R> operator>=(
    L lhs, R rhs) noexcept {
    return datapar::cmpge(lhs, rhs);
}

} // namespace operators
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
