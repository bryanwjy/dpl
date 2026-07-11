// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_trivially_copyable)
template <typename T>
inline constexpr bool is_trivially_copyable_v = __is_trivially_copyable(T);
template <typename T>
struct is_trivially_copyable : bool_constant<__is_trivially_copyable(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_trivially_copyable)
template <typename T>
inline constexpr bool is_trivially_copyable_v =
    details::type_traits::unsupported_trait_v<T>;
template <typename T>
struct is_trivially_copyable : details::type_traits::unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_volatile)

__DPL_DEFAULT_NAMESPACE_END
