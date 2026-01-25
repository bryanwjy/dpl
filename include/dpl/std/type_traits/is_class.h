// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_class)
DPL_EXPORT template <typename T>
inline constexpr bool is_class_v = __is_class(T);
DPL_EXPORT template <typename T>
struct is_class : bool_constant<__is_class(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_class)
DPL_EXPORT template <typename T>
inline constexpr bool is_class_v = unsupported_trait_v<T>;
DPL_EXPORT template <typename T>
struct is_class : unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_class)

DPL_DEFAULT_NAMESPACE_END
