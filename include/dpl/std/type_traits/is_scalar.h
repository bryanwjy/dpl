// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/is_arithmetic.h"
#include "dpl/std/type_traits/is_enum.h"
#include "dpl/std/type_traits/is_member_pointer.h"
#include "dpl/std/type_traits/is_null_pointer.h"
#include "dpl/std/type_traits/is_pointer.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_scalar)
DPL_EXPORT template <typename T>
inline constexpr bool is_scalar_v = __is_scalar(T);
DPL_EXPORT template <typename T>
struct is_scalar : bool_constant<__is_scalar(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_scalar)
DPL_EXPORT template <typename T>
inline constexpr bool is_scalar_v = is_arithmetic_v<T> || is_enum_v<T> ||
    is_member_pointer_v<T> || is_null_pointer_v<T> || is_pointer_v<T>;

DPL_EXPORT template <typename T>
struct is_scalar : bool_constant<is_scalar_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_scalar)

DPL_DEFAULT_NAMESPACE_END
