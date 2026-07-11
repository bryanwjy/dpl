// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
// IWYU pragma: begin_exports
#include "dpl/std/type_traits/is_floating_point.h"
#include "dpl/std/type_traits/is_integral.h"
// IWYU pragma: end_exports

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)
template <typename T>
inline constexpr bool is_arithmetic_v = __is_arithmetic(T);
template <typename T>
struct is_arithmetic : bool_constant<is_arithmetic_v<T>> {};

#  if DPL_SUPPORTS_EXT_FLOAT16
template <>
inline constexpr bool is_arithmetic_v<float16> = true;
template <>
inline constexpr bool is_arithmetic_v<float16 const> = true;
template <>
inline constexpr bool is_arithmetic_v<float16 volatile> = true;
template <>
inline constexpr bool is_arithmetic_v<float16 const volatile> = true;
#  endif

#  if DPL_SUPPORTS_EXT_BFLOAT16
template <>
inline constexpr bool is_arithmetic_v<bfloat16> = true;
template <>
inline constexpr bool is_arithmetic_v<bfloat16 const> = true;
template <>
inline constexpr bool is_arithmetic_v<bfloat16 volatile> = true;
template <>
inline constexpr bool is_arithmetic_v<bfloat16 const volatile> = true;
#  endif

#  if DPL_SUPPORTS_EXT_INT128
template <>
inline constexpr bool is_arithmetic_v<int128> = true;
template <>
inline constexpr bool is_arithmetic_v<int128 const> = true;
template <>
inline constexpr bool is_arithmetic_v<int128 volatile> = true;
template <>
inline constexpr bool is_arithmetic_v<int128 const volatile> = true;

template <>
inline constexpr bool is_arithmetic_v<uint128> = true;
template <>
inline constexpr bool is_arithmetic_v<uint128 const> = true;
template <>
inline constexpr bool is_arithmetic_v<uint128 volatile> = true;
template <>
inline constexpr bool is_arithmetic_v<uint128 const volatile> = true;
#  endif

#else  // if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)
template <typename T>
inline constexpr bool is_arithmetic_v =
    is_integral_v<T> || is_floating_point_v<T>;

template <typename T>
struct is_arithmetic : bool_constant<is_arithmetic_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)

__DPL_DEFAULT_NAMESPACE_END
