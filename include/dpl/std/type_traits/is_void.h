// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_void)
DPL_EXPORT template <typename T>
inline constexpr bool is_void_v = __is_void(T);
DPL_EXPORT template <typename T>
struct is_void : bool_constant<__is_void(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_void)
DPL_EXPORT template <typename T>
inline constexpr bool is_void_v = false;
DPL_EXPORT template <>
inline constexpr bool is_void_v<void> = true;
DPL_EXPORT template <>
inline constexpr bool is_void_v<void const> = true;
DPL_EXPORT template <>
inline constexpr bool is_void_v<void volatile> = true;
DPL_EXPORT template <>
inline constexpr bool is_void_v<void const volatile> = true;
DPL_EXPORT template <typename T>
struct is_void : bool_constant<is_void_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_void)

DPL_DEFAULT_NAMESPACE_END
