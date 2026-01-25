// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_const)
DPL_EXPORT template <typename T>
inline constexpr bool is_const_v = __is_const(T);
DPL_EXPORT template <typename T>
struct is_const : bool_constant<__is_const(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_const)
DPL_EXPORT template <typename T>
inline constexpr bool is_const_v = false;
DPL_EXPORT template <typename T>
inline constexpr bool is_const_v<T const> = true;
DPL_EXPORT template <typename T>
struct is_const : bool_constant<is_const_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_const)

DPL_DEFAULT_NAMESPACE_END
