// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_nothrow_destructible)
template <typename T>
inline constexpr bool is_nothrow_destructible_v = __is_nothrow_destructible(T);
template <typename T>
struct is_nothrow_destructible : bool_constant<__is_nothrow_destructible(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_nothrow_destructible)
template <typename T>
inline constexpr bool is_nothrow_destructible_v = requires(T* ptr) {
    { ptr->~T() } noexcept;
};
template <typename T>
inline constexpr bool is_nothrow_destructible_v<T const> =
    is_nothrow_destructible_v<T>;
template <typename T>
inline constexpr bool is_nothrow_destructible_v<T volatile> =
    is_nothrow_destructible_v<T>;
template <typename T>
inline constexpr bool is_nothrow_destructible_v<T const volatile> =
    is_nothrow_destructible_v<T>;

template <typename T, typename... Args>
struct is_nothrow_destructible :
    bool_constant<is_nothrow_destructible_v<T, Args...>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_nothrow_destructible)

__DPL_DEFAULT_NAMESPACE_END
