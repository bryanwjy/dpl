// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_pointer)
DPL_EXPORT template <typename T>
inline constexpr bool is_pointer_v = __is_pointer(T);
DPL_EXPORT template <typename T>
struct is_pointer : bool_constant<__is_pointer(T)> {};
#else  // __DPL_SHOULD_USE_BUILTIN(is_pointer)
DPL_EXPORT template <typename T>
inline constexpr bool is_pointer_v = false;
DPL_EXPORT template <typename T>
inline constexpr bool is_pointer_v<T*> = true;
DPL_EXPORT template <typename T>
inline constexpr bool is_pointer_v<T* const> = true;
DPL_EXPORT template <typename T>
inline constexpr bool is_pointer_v<T* volatile> = true;
DPL_EXPORT template <typename T>
inline constexpr bool is_pointer_v<T* const volatile> = true;
DPL_EXPORT template <typename T>
struct is_pointer : bool_constant<is_pointer_v<T>> {};
#endif // __DPL_SHOULD_USE_BUILTIN(is_pointer)

DPL_DEFAULT_NAMESPACE_END
