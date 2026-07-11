// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/is_array.h"
#include "dpl/std/type_traits/is_class.h"
#include "dpl/std/type_traits/is_scalar.h"
#include "dpl/std/type_traits/is_union.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_object)
template <typename T>
inline constexpr bool is_object_v = __is_object(T);
template <typename T>
struct is_object : bool_constant<__is_object(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_object)
template <typename T>
inline constexpr bool is_object_v =
    is_scalar_v<T> || is_array_v<T> || is_union_v<T> || is_class_v<T>;

template <typename T>
struct is_object : bool_constant<is_object_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_object)

__DPL_DEFAULT_NAMESPACE_END
