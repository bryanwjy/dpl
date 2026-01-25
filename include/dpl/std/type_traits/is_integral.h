// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_integral)
DPL_EXPORT template <typename T>
inline constexpr bool is_integral_v = __is_integral(T);
DPL_EXPORT template <typename T>
struct is_integral : bool_constant<__is_integral(T)> {};
#else // if __DPL_SHOULD_USE_BUILTIN(is_integral)
DPL_EXPORT template <typename T>
inline constexpr bool is_integral_v = false;
DPL_EXPORT template <typename T>
inline constexpr bool is_integral_v<T const> = is_integral_v<T>;
DPL_EXPORT template <typename T>
inline constexpr bool is_integral_v<T volatile> = is_integral_v<T>;
DPL_EXPORT template <typename T>
inline constexpr bool is_integral_v<T const volatile> = is_integral_v<T>;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<bool> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<char> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<signed char> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<unsigned char> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<wchar_t> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<char16_t> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<char32_t> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<short> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<unsigned short> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<int> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<unsigned int> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<long> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<unsigned long> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<long long> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<unsigned long long> = true;
#  if DPL_SUPPORTS_CHAR8_T
DPL_EXPORT template <>
inline constexpr bool is_integral_v<char8_t> = true;
#  endif // if DPL_SUPPORTS_CHAR8_T
#  if DPL_SUPPORTS_INT128
DPL_EXPORT template <>
inline constexpr bool is_integral_v<__int128_t> = true;
DPL_EXPORT template <>
inline constexpr bool is_integral_v<__uint128_t> = true;
#  endif // if DPL_SUPPORTS_INT128
DPL_EXPORT template <typename T>
struct is_integral : bool_constant<is_integral_v<T>> {};
#endif   // if __DPL_SHOULD_USE_BUILTIN(is_integral)

DPL_DEFAULT_NAMESPACE_END
