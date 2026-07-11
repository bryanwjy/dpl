// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/is_const.h"
#include "dpl/std/type_traits/is_reference.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_function)
template <typename T>
inline constexpr bool is_function_v = __is_function(T);
template <typename T>
struct is_function : bool_constant<__is_function(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_function)
template <typename T>
inline constexpr bool is_function_v =
    !is_const_v<T const> && !is_reference_v<T>;

template <typename T>
struct is_function : bool_constant<is_function_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_function)

__DPL_DEFAULT_NAMESPACE_END
