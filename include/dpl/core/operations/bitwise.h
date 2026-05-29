// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/std/type_traits/enable_if.h"
#endif

// IWYU pragma: begin_exports
#include "dpl/core/operations/bitwise/bwand.h"
#include "dpl/core/operations/bitwise/bwandnot.h"
#include "dpl/core/operations/bitwise/bwnot.h"
#include "dpl/core/operations/bitwise/bwor.h"
#include "dpl/core/operations/bitwise/bwornot.h"
#include "dpl/core/operations/bitwise/bwshift_left.h"
#include "dpl/core/operations/bitwise/bwshift_right.h"
#include "dpl/core/operations/bitwise/bwxor.h"
// IWYU pragma: end_exports

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename D>
class bitwise_simd_interface {
public:
    template <typename R>
    requires regular_invocable<internal::bwor_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::bwor_t, D, R> operator|(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::bwor(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwand_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::bwand_t, D, R> operator&(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::bwand(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwxor_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::bwxor_t, D, R> operator^(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::bwxor(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwshift_left_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::bwshift_left_t, D, R> operator<<(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::bwshift_left(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwshift_right_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::bwshift_right_t, D, R> operator>>(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::bwshift_right(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwor_t, D, R> &&
        assignable_from<D&, invoke_result_t<internal::bwor_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator|=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::bwor(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwand_t, D, R> &&
        assignable_from<D&, invoke_result_t<internal::bwand_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator&=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::bwand(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwxor_t, D, R> &&
        assignable_from<D&, invoke_result_t<internal::bwxor_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator^=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::bwxor(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwshift_left_t, D, R> &&
        assignable_from<D&, invoke_result_t<internal::bwshift_left_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator<<=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::bwshift_left(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::bwshift_right_t, D, R> &&
        assignable_from<D&, invoke_result_t<internal::bwshift_right_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator>>=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::bwshift_right(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::bwor_t, L, D> operator|(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::bwor(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::bwand_t, L, D> operator&(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::bwand(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::bwxor_t, L, D> operator^(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::bwxor(lhs, rhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr D operator~(this D self) noexcept
    requires simd_vector<D> && regular_invocable<internal::bwnot_t, D>
    {
        return datapar::bwnot(self);
    }
};

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwor_t, L, R> operator|(
    L lhs, R rhs) noexcept {
    return datapar::bwor(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwand_t, L, R> operator&(
    L lhs, R rhs) noexcept {
    return datapar::bwand(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwxor_t, L, R> operator^(
    L lhs, R rhs) noexcept {
    return datapar::bwxor(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwshift_left_t, L, R> operator<<(
    L lhs, R rhs) noexcept {
    return datapar::bwshift_left(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::bwshift_right_t, L, R> operator>>(
    L lhs, R rhs) noexcept {
    return datapar::bwshift_right(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
requires requires {
    typename invoke_result_t<internal::bwor_t, L, R>;
    requires assignable_from<L&, invoke_result_t<internal::bwor_t, L, R>>;
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr L& operator|=(L& lhs, R rhs) noexcept {
    return lhs = lhs | rhs;
}

DPL_EXPORT template <typename L, typename R>
requires requires {
    typename invoke_result_t<internal::bwand_t, L, R>;
    requires assignable_from<L&, invoke_result_t<internal::bwand_t, L, R>>;
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr L& operator&=(L& lhs, R rhs) noexcept {
    return lhs = lhs & rhs;
}

DPL_EXPORT template <typename L, typename R>
requires requires {
    typename invoke_result_t<internal::bwxor_t, L, R>;
    requires assignable_from<L&, invoke_result_t<internal::bwxor_t, L, R>>;
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr L& operator^=(L& lhs, R rhs) noexcept {
    return lhs = lhs ^ rhs;
}

DPL_EXPORT template <typename L, typename R>
requires requires {
    typename invoke_result_t<internal::bwshift_left_t, L, R>;
    requires assignable_from<L&,
        invoke_result_t<internal::bwshift_left_t, L, R>>;
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr L& operator<<=(L& lhs, R rhs) noexcept {
    return lhs = (lhs << rhs);
}

DPL_EXPORT template <typename L, typename R>
requires requires {
    typename invoke_result_t<internal::bwshift_right_t, L, R>;
    requires assignable_from<L&,
        invoke_result_t<internal::bwshift_right_t, L, R>>;
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr L& operator&=(L& lhs, R rhs) noexcept {
    return lhs = (lhs >> rhs);
}

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto operator~(T val) noexcept
    -> invoke_result_t<internal::bwnot_t, T> {
    return datapar::bwnot(val);
}

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
