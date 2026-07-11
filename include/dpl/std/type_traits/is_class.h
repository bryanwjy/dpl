// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_class)
template <typename T>
inline constexpr bool is_class_v = __is_class(T);
template <typename T>
struct is_class : bool_constant<__is_class(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_class)
template <typename T>
inline constexpr bool is_class_v = details::type_traits::unsupported_trait_v<T>;
template <typename T>
struct is_class : details::type_traits::unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_class)

__DPL_DEFAULT_NAMESPACE_END
