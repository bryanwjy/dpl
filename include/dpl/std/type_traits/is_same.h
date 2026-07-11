// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_same)
template <typename T, typename U>
inline constexpr bool is_same_v = __is_same(T, U);
template <typename T, typename U>
struct is_same : bool_constant<__is_same(T, U)> {};

#else  // if __DPL_SHOULD_USE_BUILTIN(is_same)
template <typename, typename>
inline constexpr bool is_same_v = false;
template <typename T>
inline constexpr bool is_same_v<T, T> = true;
template <typename T>
struct is_same : bool_constant<is_same_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_same)

__DPL_DEFAULT_NAMESPACE_END
