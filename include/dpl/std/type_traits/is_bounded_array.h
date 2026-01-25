// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_bounded_array)
DPL_EXPORT template <typename T>
inline constexpr bool is_bounded_array_v = __is_bounded_array(T);
DPL_EXPORT template <typename T>
struct is_bounded_array : bool_constant<__is_bounded_array(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_bounded_array)
DPL_EXPORT template <typename T>
inline constexpr bool is_bounded_array_v = false;
DPL_EXPORT template <typename T, size_t N>
inline constexpr bool is_bounded_array_v<T[N]> = true;
DPL_EXPORT template <typename T>
struct is_bounded_array : bool_constant<is_bounded_array_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_bounded_array)

DPL_DEFAULT_NAMESPACE_END
