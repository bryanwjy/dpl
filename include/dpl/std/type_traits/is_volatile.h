// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_volatile)
DPL_EXPORT template <typename T>
inline constexpr bool is_volatile_v = __is_volatile(T);
DPL_EXPORT template <typename T>
struct is_volatile : bool_constant<__is_volatile(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_volatile)
DPL_EXPORT template <typename T>
inline constexpr bool is_volatile_v = false;
DPL_EXPORT template <typename T>
inline constexpr bool is_volatile_v<T volatile> = true;
DPL_EXPORT template <typename T>
struct is_volatile : bool_constant<is_volatile_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_volatile)

DPL_DEFAULT_NAMESPACE_END
