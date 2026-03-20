// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_floating_point)
DPL_EXPORT template <typename T>
inline constexpr bool is_floating_point_v = __is_floating_point(T);
#  if DPL_SUPPORTS_EXT_FLOAT16
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<float16> = true;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<float16 const> = true;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<float16 const volatile> = true;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<float16 volatile> = true;
#  endif
#  if DPL_SUPPORTS_EXT_BFLOAT16
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<bfloat16> = true;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<bfloat16 const> = true;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<bfloat16 const volatile> = true;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<bfloat16 volatile> = true;
#  endif
DPL_EXPORT template <typename T>
struct is_floating_point : bool_constant<is_floating_point_v<T>> {};
#else
DPL_EXPORT template <typename T>
inline constexpr bool is_floating_point_v = false;
DPL_EXPORT template <typename T>
inline constexpr bool is_floating_point_v<T const> = is_floating_point_v<T>;
DPL_EXPORT template <typename T>
inline constexpr bool is_floating_point_v<T volatile> = is_floating_point_v<T>;
DPL_EXPORT template <typename T>
inline constexpr bool is_floating_point_v<T const volatile> =
    is_floating_point_v<T>;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<float> = true;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<double> = true;
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<long double> = true;

#  if DPL_SUPPORTS_FLOAT16
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<decltype(0.0f16)> = true;
#  endif // if DPL_SUPPORTS_FLOAT16
#  if DPL_SUPPORTS_FLOAT32
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<decltype(0.0f32)> = true;
#  endif // if DPL_SUPPORTS_FLOAT32
#  if DPL_SUPPORTS_FLOAT64
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<decltype(0.0f64)> = true;
#  endif // if DPL_SUPPORTS_FLOAT64
#  if DPL_SUPPORTS_FLOAT128
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<decltype(0.0f128)> = true;
#  endif // if DPL_SUPPORTS_FLOAT128
#  if DPL_SUPPORTS_BFLOAT16
DPL_EXPORT template <>
inline constexpr bool is_floating_point_v<decltype(0.0bf16)> = true;
#  endif // if DPL_SUPPORTS_BFLOAT16
DPL_EXPORT template <typename T>
struct is_floating_point : bool_constant<is_floating_point_v<T>> {};
#endif

DPL_DEFAULT_NAMESPACE_END
