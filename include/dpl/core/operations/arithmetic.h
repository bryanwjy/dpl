// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/type_traits/enable_if.h"
#endif

// IWYU pragma: begin_exports
#include "dpl/core/operations/arithmetic/abs.h"
#include "dpl/core/operations/arithmetic/add.h"
#include "dpl/core/operations/arithmetic/addsub.h"
#include "dpl/core/operations/arithmetic/divide.h"
#include "dpl/core/operations/arithmetic/fmadd.h"
#include "dpl/core/operations/arithmetic/fmaddsub.h"
#include "dpl/core/operations/arithmetic/fmsub.h"
#include "dpl/core/operations/arithmetic/fmsubadd.h"
#include "dpl/core/operations/arithmetic/fnmadd.h"
#include "dpl/core/operations/arithmetic/fnmsub.h"
#include "dpl/core/operations/arithmetic/multiply.h"
#include "dpl/core/operations/arithmetic/negate.h"
#include "dpl/core/operations/arithmetic/result.h"
#include "dpl/core/operations/arithmetic/subadd.h"
#include "dpl/core/operations/arithmetic/subtract.h"
// IWYU pragma: end_exports

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename D>
class arithmetic_vector_interface {
public:
    template <typename R>
    requires regular_invocable<internal::add_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::add_t, D, R> operator+(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::add(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::subtract_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::subtract_t, D, R> operator-(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::subtract(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::multiply_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::multiply_t, D, R> operator*(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::multiply(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::divide_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::divide_t, D, R> operator/(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::divide(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::add_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::add_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator+=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::add(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::subtract_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::subtract_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator-=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::subtract(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::multiply_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::multiply_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator*=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::multiply(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::divide_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::divide_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator/=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::divide(lhs, rhs));
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::add_t, L, D> operator+(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::add(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::subtract_t, L, D> operator-(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::subtract(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::multiply_t, L, D> operator*(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::multiply(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::divide_t, L, D> operator/(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::divide(lhs, rhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr D operator-(this D self) noexcept
    requires simd_vector<D> && basic_element<simd_element_type_t<D>> &&
        regular_invocable<internal::negate_t, D>
    {
        return internal::negate_t::operator()(self);
    }
};

/**
 * The following are not exported by design
 */
DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::add_t, L, R> operator+(
    L lhs, R rhs) noexcept {
    return datapar::add(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::subtract_t, L, R> operator-(
    L lhs, R rhs) noexcept {
    return datapar::subtract(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::multiply_t, L, R> operator*(
    L lhs, R rhs) noexcept {
    return datapar::multiply(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::divide_t, L, R> operator/(
    L lhs, R rhs) noexcept {
    return datapar::divide(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator+=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_type_with<L, decltype(lhs + rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs + rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator-=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_type_with<L, decltype(lhs - rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs - rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator*=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_type_with<L, decltype(lhs * rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs * rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator/=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_type_with<L, decltype(lhs / rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs / rhs);
}

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto operator-(T val) noexcept
    -> invoke_result_t<internal::negate_t, T> {
    return internal::negate_t::operator()(val);
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
