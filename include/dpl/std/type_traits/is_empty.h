// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_empty)
DPL_EXPORT template <typename T>
inline constexpr bool is_empty_v = __is_empty(T);
DPL_EXPORT template <typename T>
struct is_empty : bool_constant<__is_empty(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_empty)
DPL_EXPORT template <typename T>
inline constexpr bool is_empty_v = details::type_traits::unsupported_trait_v<T>;
DPL_EXPORT template <typename T>
struct is_empty : details::type_traits::unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_empty)

DPL_DEFAULT_NAMESPACE_END
