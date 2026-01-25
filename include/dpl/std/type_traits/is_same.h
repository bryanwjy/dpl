// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_same)
DPL_EXPORT template <typename T, typename U>
inline constexpr bool is_same_v = __is_same(T, U);
DPL_EXPORT template <typename T, typename U>
struct is_same : bool_constant<__is_same(T, U)> {};

#else  // if __DPL_SHOULD_USE_BUILTIN(is_same)
DPL_EXPORT template <typename, typename>
inline constexpr bool is_same_v = false;
DPL_EXPORT template <typename T>
inline constexpr bool is_same_v<T, T> = true;
DPL_EXPORT template <typename T>
struct is_same : bool_constant<is_same_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_same)

DPL_DEFAULT_NAMESPACE_END
