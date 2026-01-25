// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/as_const.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_constructible.h"
#  include "dpl/std/type_traits/is_lvalue_reference.h"
#  include "dpl/std/type_traits/is_nothrow_constructible.h"
#  include "dpl/std/type_traits/remove_reference.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(NODISCARD, ALWAYS_INLINE, _HIDE_FROM_ABI)
constexpr remove_reference_t<T>&& move(
    T&& ref DPL_LIFETIMEBOUND) noexcept {
    return static_cast<remove_reference_t<T>&&>(ref);
}

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(NODISCARD, ALWAYS_INLINE, _HIDE_FROM_ABI)
constexpr decltype(auto)
    move_if_noexcept(T& ref DPL_LIFETIMEBOUND) noexcept {
    if constexpr (!is_nothrow_move_constructible_v<T> &&
        is_copy_constructible_v<T>) {
        return __DPL as_const(ref);
    } else {
        return __DPL move(ref);
    }
}

DPL_DEFAULT_NAMESPACE_END
