// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#if !__DPL_SHOULD_USE_BUILTIN(is_convertible)
#  include "dpl/std/type_traits/is_void.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_convertible)
template <typename From, typename To>
inline constexpr bool is_convertible_v = __is_convertible(From, To);
template <typename From, typename To>
struct is_convertible : bool_constant<__is_convertible(From, To)> {};

#else // if __DPL_SHOULD_USE_BUILTIN(is_convertible)

template <typename From, typename To>
inline constexpr bool is_convertible_v = is_void_v<From> && is_void_v<To> ||
    requires(void (*to)(To), From (*from)()) { to(from()); };

template <typename From, typename To>
struct is_convertible : bool_constant<is_convertible_v<From, To>> {};

#endif // if __DPL_SHOULD_USE_BUILTIN(is_convertible)

template <typename From, typename To>
inline constexpr bool is_core_convertible_v =
    requires(void (*to)(To), From (*from)()) { to(from()); };

template <typename From, typename To>
struct is_core_convertible : bool_constant<is_core_convertible_v<From, To>> {};

__DPL_DEFAULT_NAMESPACE_END
