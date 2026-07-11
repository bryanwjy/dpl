// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_pointer)
template <typename T>
inline constexpr bool is_pointer_v = __is_pointer(T);
template <typename T>
struct is_pointer : bool_constant<__is_pointer(T)> {};
#else  // __DPL_SHOULD_USE_BUILTIN(is_pointer)
template <typename T>
inline constexpr bool is_pointer_v = false;
template <typename T>
inline constexpr bool is_pointer_v<T*> = true;
template <typename T>
inline constexpr bool is_pointer_v<T* const> = true;
template <typename T>
inline constexpr bool is_pointer_v<T* volatile> = true;
template <typename T>
inline constexpr bool is_pointer_v<T* const volatile> = true;
template <typename T>
struct is_pointer : bool_constant<is_pointer_v<T>> {};
#endif // __DPL_SHOULD_USE_BUILTIN(is_pointer)

__DPL_DEFAULT_NAMESPACE_END
