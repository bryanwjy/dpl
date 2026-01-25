// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/is_floating_point.h"
#include "dpl/std/type_traits/is_integral.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)
DPL_EXPORT template <typename T>
inline constexpr bool is_arithmetic_v = __is_arithmetic(T);
DPL_EXPORT template <typename T>
struct is_arithmetic : bool_constant<__is_arithmetic(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)
DPL_EXPORT template <typename T>
inline constexpr bool is_arithmetic_v =
    is_integral_v<T> || is_floating_point_v<T>;

DPL_EXPORT template <typename T>
struct is_arithmetic : bool_constant<is_arithmetic_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_arithmetic)

DPL_DEFAULT_NAMESPACE_END
