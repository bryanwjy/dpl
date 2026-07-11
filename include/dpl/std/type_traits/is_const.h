// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_const)
template <typename T>
inline constexpr bool is_const_v = __is_const(T);
template <typename T>
struct is_const : bool_constant<__is_const(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_const)
template <typename T>
inline constexpr bool is_const_v = false;
template <typename T>
inline constexpr bool is_const_v<T const> = true;
template <typename T>
struct is_const : bool_constant<is_const_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_const)

__DPL_DEFAULT_NAMESPACE_END
