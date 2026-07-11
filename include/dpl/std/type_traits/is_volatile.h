// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_volatile)
template <typename T>
inline constexpr bool is_volatile_v = __is_volatile(T);
template <typename T>
struct is_volatile : bool_constant<__is_volatile(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_volatile)
template <typename T>
inline constexpr bool is_volatile_v = false;
template <typename T>
inline constexpr bool is_volatile_v<T volatile> = true;
template <typename T>
struct is_volatile : bool_constant<is_volatile_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_volatile)

__DPL_DEFAULT_NAMESPACE_END
