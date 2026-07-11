// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_aggregate)
template <typename T>
inline constexpr bool is_aggregate_v = __is_aggregate(T);
template <typename T>
struct is_aggregate : bool_constant<__is_aggregate(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)
template <typename T>
inline constexpr bool is_aggregate_v =
    details::type_traits::unsupported_trait_v<T>;
template <typename T>
struct is_aggregate : details::type_traits::unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

__DPL_DEFAULT_NAMESPACE_END
