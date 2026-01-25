// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#if !__DPL_SHOULD_USE_BUILTIN(is_nothrow_convertible)
#  include "dpl/std/type_traits/is_void.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_nothrow_convertible)
DPL_EXPORT template <typename From, typename To>
inline constexpr bool is_nothrow_convertible_v =
    __is_nothrow_convertible(From, To);
DPL_EXPORT template <typename From, typename To>
struct is_nothrow_convertible :
    bool_constant<__is_nothrow_convertible(From, To)> {};
#else // if __DPL_SHOULD_USE_BUILTIN(is_nothrow_convertible)
DPL_EXPORT template <typename From, typename To>
inline constexpr bool is_nothrow_convertible_v =
    is_void_v<From> && is_void_v<To> ||
    requires(void (*to)(To) noexcept, From (*from)() noexcept) {
        { to(from()) } noexcept;
    };

DPL_EXPORT template <typename From, typename To>
struct is_convertible : bool_constant<is_nothrow_convertible_v<From, To>> {};

#endif // if __DPL_SHOULD_USE_BUILTIN(is_nothrow_convertible)

DPL_DEFAULT_NAMESPACE_END
