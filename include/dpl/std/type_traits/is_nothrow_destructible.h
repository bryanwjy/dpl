// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_nothrow_destructible)
DPL_EXPORT template <typename T>
inline constexpr bool is_nothrow_destructible_v = __is_nothrow_destructible(T);
DPL_EXPORT template <typename T>
struct is_nothrow_destructible : bool_constant<__is_nothrow_destructible(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_nothrow_destructible)
DPL_EXPORT template <typename T>
inline constexpr bool is_nothrow_destructible_v = requires(T* ptr) {
    { ptr->~T() } noexcept;
};
DPL_EXPORT template <typename T>
inline constexpr bool is_nothrow_destructible_v<T const> =
    is_nothrow_destructible_v<T>;
DPL_EXPORT template <typename T>
inline constexpr bool is_nothrow_destructible_v<T volatile> =
    is_nothrow_destructible_v<T>;
DPL_EXPORT template <typename T>
inline constexpr bool is_nothrow_destructible_v<T const volatile> =
    is_nothrow_destructible_v<T>;

DPL_EXPORT template <typename T, typename... Args>
struct is_nothrow_destructible :
    bool_constant<is_nothrow_destructible_v<T, Args...>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_nothrow_destructible)

DPL_DEFAULT_NAMESPACE_END
