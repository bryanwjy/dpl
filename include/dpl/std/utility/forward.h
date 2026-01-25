// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_lvalue_reference.h"
#  include "dpl/std/type_traits/remove_reference.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(NODISCARD, ALWAYS_INLINE, _HIDE_FROM_ABI)
constexpr T&& forward(
    remove_reference_t<T>& ref DPL_LIFETIMEBOUND) noexcept {
    return static_cast<T&&>(ref);
}

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(NODISCARD, ALWAYS_INLINE, _HIDE_FROM_ABI)
constexpr T&& forward(
    remove_reference_t<T>&& ref DPL_LIFETIMEBOUND) noexcept {
    static_assert(!is_lvalue_reference<T>::value,
        "cannot forward an rvalue as an lvalue");
    return static_cast<T&&>(ref);
}

DPL_DEFAULT_NAMESPACE_END
