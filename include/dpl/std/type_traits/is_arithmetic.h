// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
// IWYU pragma: begin_exports
#include "dpl/std/type_traits/is_floating_point.h"
#include "dpl/std/type_traits/is_integral.h"
// IWYU pragma: end_exports

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)
DPL_EXPORT template <typename T>
inline constexpr bool is_arithmetic_v = __is_arithmetic(T);
#  if DPL_SUPPORTS_EXT_FLOAT16
DPL_EXPORT template <>
inline constexpr bool is_arithmetic_v<float16> = true;
DPL_EXPORT template <>
inline constexpr bool is_arithmetic_v<float16 const> = true;
DPL_EXPORT template <>
inline constexpr bool is_arithmetic_v<float16 const volatile> = true;
DPL_EXPORT template <>
inline constexpr bool is_arithmetic_v<float16 volatile> = true;
#  endif
#  if DPL_SUPPORTS_EXT_BFLOAT16
DPL_EXPORT template <>
inline constexpr bool is_arithmetic_v<bfloat16> = true;
DPL_EXPORT template <>
inline constexpr bool is_arithmetic_v<bfloat16 const> = true;
DPL_EXPORT template <>
inline constexpr bool is_arithmetic_v<bfloat16 const volatile> = true;
DPL_EXPORT template <>
inline constexpr bool is_arithmetic_v<bfloat16 volatile> = true;
#  endif
DPL_EXPORT template <typename T>
struct is_arithmetic : bool_constant<is_arithmetic_v<T>> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)
DPL_EXPORT template <typename T>
inline constexpr bool is_arithmetic_v =
    is_integral_v<T> || is_floating_point_v<T>;

DPL_EXPORT template <typename T>
struct is_arithmetic : bool_constant<is_arithmetic_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)

DPL_DEFAULT_NAMESPACE_END
