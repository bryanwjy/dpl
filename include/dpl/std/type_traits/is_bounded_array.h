// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_bounded_array)
template <typename T>
inline constexpr bool is_bounded_array_v = __is_bounded_array(T);
template <typename T>
struct is_bounded_array : bool_constant<__is_bounded_array(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_bounded_array)
template <typename T>
inline constexpr bool is_bounded_array_v = false;
template <typename T, size_t N>
inline constexpr bool is_bounded_array_v<T[N]> = true;
template <typename T>
struct is_bounded_array : bool_constant<is_bounded_array_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_bounded_array)

__DPL_DEFAULT_NAMESPACE_END
