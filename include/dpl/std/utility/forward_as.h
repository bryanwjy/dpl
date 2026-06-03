// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/as_const.h"
#include "dpl/std/utility/move.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/forwardable_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename From, typename To>
requires convertible_to<remove_reference_t<From>&&, To&&>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr copy_cvref_t<remove_reference_t<From>&&, To> forward_as(
    remove_reference_t<From>&& x DPL_LIFETIMEBOUND) noexcept {
    return static_cast<copy_cvref_t<remove_reference_t<From>&&, To>>(
        static_cast<From&&>(x));
}

DPL_EXPORT template <typename From, typename To>
requires convertible_to<remove_reference_t<From>&, To&>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr copy_cvref_t<remove_reference_t<From>&, To> forward_as(
    remove_reference_t<From>& x DPL_LIFETIMEBOUND) noexcept {
    return static_cast<copy_cvref_t<remove_reference_t<From>&, To>>(x);
}

DPL_DEFAULT_NAMESPACE_END
