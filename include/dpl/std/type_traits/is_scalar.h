// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
// IWYU pragma: begin_exports
#include "dpl/std/type_traits/is_arithmetic.h"
#include "dpl/std/type_traits/is_enum.h"
#include "dpl/std/type_traits/is_member_pointer.h"
#include "dpl/std/type_traits/is_null_pointer.h"
#include "dpl/std/type_traits/is_pointer.h"
// IWYU pragma: end_exports

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_scalar)
template <typename T>
inline constexpr bool is_scalar_v = __is_scalar(T);
template <typename T>
struct is_scalar : bool_constant<is_scalar_v<T>> {};

#  if DPL_SUPPORTS_EXT_FLOAT16
template <>
inline constexpr bool is_scalar_v<float16> = true;
template <>
inline constexpr bool is_scalar_v<float16 const> = true;
template <>
inline constexpr bool is_scalar_v<float16 volatile> = true;
template <>
inline constexpr bool is_scalar_v<float16 const volatile> = true;
#  endif

#  if DPL_SUPPORTS_EXT_BFLOAT16
template <>
inline constexpr bool is_scalar_v<bfloat16> = true;
template <>
inline constexpr bool is_scalar_v<bfloat16 const> = true;
template <>
inline constexpr bool is_scalar_v<bfloat16 volatile> = true;
template <>
inline constexpr bool is_scalar_v<bfloat16 const volatile> = true;
#  endif

#  if DPL_SUPPORTS_INT128
template <>
inline constexpr bool is_scalar_v<int128> = true;
template <>
inline constexpr bool is_scalar_v<int128 const> = true;
template <>
inline constexpr bool is_scalar_v<int128 volatile> = true;
template <>
inline constexpr bool is_scalar_v<int128 const volatile> = true;

template <>
inline constexpr bool is_scalar_v<uint128> = true;
template <>
inline constexpr bool is_scalar_v<uint128 const> = true;
template <>
inline constexpr bool is_scalar_v<uint128 volatile> = true;
template <>
inline constexpr bool is_scalar_v<uint128 const volatile> = true;
#  endif

#else  // if __DPL_SHOULD_USE_BUILTIN(is_scalar)
template <typename T>
inline constexpr bool is_scalar_v = is_arithmetic_v<T> || is_enum_v<T> ||
    is_member_pointer_v<T> || is_null_pointer_v<T> || is_pointer_v<T>;

template <typename T>
struct is_scalar : bool_constant<is_scalar_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_scalar)

__DPL_DEFAULT_NAMESPACE_END
