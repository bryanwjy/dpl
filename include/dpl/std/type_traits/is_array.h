// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

/**
 * Note that clang at some point had a bug where __is_array(T[0]) evaluates to
 * true which is incorrect, but this has since been fixed.
 */
#define DPL_DISABLE_BUILTIN_is_array (!DPL_COMPILER_CLANG_AT_LEAST(19, 00, 00))

#if __DPL_SHOULD_USE_BUILTIN(is_array)
template <typename T>
inline constexpr bool is_array_v = __is_array(T);
template <typename T>
struct is_array : bool_constant<__is_array(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_array)
template <typename T>
inline constexpr bool is_array_v = false;
template <typename T>
inline constexpr bool is_array_v<T[]> = true;
template <typename T, size_t N>
inline constexpr bool is_array_v<T[N]> = true;
template <typename T>
struct is_array : bool_constant<is_array_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_array)

__DPL_DEFAULT_NAMESPACE_END
