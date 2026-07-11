// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_trivially_destructible)
template <typename T>
inline constexpr bool is_trivially_destructible_v =
    __is_trivially_destructible(T);
template <typename T>
struct is_trivially_destructible :
    bool_constant<__is_trivially_destructible(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_trivially_destructible)

template <typename T>
inline constexpr bool is_trivially_destructible_v =
    details::type_traits::unsupported_trait_v<T>;
template <typename T>
inline constexpr bool is_trivially_destructible_v<T const> =
    is_trivially_destructible_v<T>;
template <typename T>
inline constexpr bool is_trivially_destructible_v<T volatile> =
    is_trivially_destructible_v<T>;
template <typename T>
inline constexpr bool is_trivially_destructible_v<T const volatile> =
    is_trivially_destructible_v<T>;

template <typename T, typename... Args>
struct is_trivially_destructible :
    bool_constant<is_trivially_destructible_v<T, Args...>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_trivially_destructible)

__DPL_DEFAULT_NAMESPACE_END
