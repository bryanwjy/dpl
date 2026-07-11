// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_null_pointer)
template <typename T>
inline constexpr bool is_null_pointer_v = __is_null_pointer(T);
template <typename T>
struct is_null_pointer : bool_constant<__is_null_pointer(T)> {};
#else  // __DPL_SHOULD_USE_BUILTIN(is_null_pointer)
template <typename T>
inline constexpr bool is_null_pointer_v = false;
template <>
inline constexpr bool is_null_pointer_v<decltype(nullptr)> = true;
template <>
inline constexpr bool is_null_pointer_v<decltype(nullptr) const> = true;
template <>
inline constexpr bool is_null_pointer_v<decltype(nullptr) volatile> = true;
template <>
inline constexpr bool is_null_pointer_v<decltype(nullptr) const volatile> =
    true;
template <typename T>
struct is_null_pointer : bool_constant<is_null_pointer_v<T>> {};
#endif // __DPL_SHOULD_USE_BUILTIN(is_null_pointer)

__DPL_DEFAULT_NAMESPACE_END
