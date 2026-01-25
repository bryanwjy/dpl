// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/is_function.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_member_function_pointer)
DPL_EXPORT template <typename T>
inline constexpr bool is_member_function_pointer_v =
    __is_member_function_pointer(T);
DPL_EXPORT template <typename T>
struct is_member_function_pointer :
    bool_constant<__is_member_function_pointer(T)> {};
#else  // __DPL_SHOULD_USE_BUILTIN(is_member_function_pointer)
DPL_EXPORT template <typename T>
inline constexpr bool is_member_function_pointer_v = false;
DPL_EXPORT template <typename T>
inline constexpr bool is_member_function_pointer_v<T const> =
    is_member_function_pointer_v<T>;
DPL_EXPORT template <typename T>
inline constexpr bool is_member_function_pointer_v<T volatile> =
    is_member_function_pointer_v<T>;
DPL_EXPORT template <typename T>
inline constexpr bool is_member_function_pointer_v<T const volatile> =
    is_member_function_pointer_v<T>;
DPL_EXPORT template <typename C, typename T>
inline constexpr bool is_member_function_pointer_v<T C::*> = is_function_v<T>;

DPL_EXPORT template <typename T>
struct is_member_function_pointer :
    bool_constant<is_member_function_pointer_v<T>> {};
#endif // __DPL_SHOULD_USE_BUILTIN(is_member_function_pointer)

DPL_DEFAULT_NAMESPACE_END
