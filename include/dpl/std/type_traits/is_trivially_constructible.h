// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/add_lvalue_reference.h"
#include "dpl/std/type_traits/add_rvalue_reference.h"
#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_trivially_constructible)
DPL_EXPORT template <typename T, typename... Args>
inline constexpr bool is_trivially_constructible_v =
    __is_trivially_constructible(T, Args...);
DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_default_constructible_v =
    __is_trivially_constructible(T);
DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_copy_constructible_v =
    __is_trivially_constructible(T, add_lvalue_reference_t<T const>);
DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_move_constructible_v =
    __is_trivially_constructible(T, add_rvalue_reference_t<T>);

DPL_EXPORT template <typename T, typename... Args>
struct is_trivially_constructible :
    bool_constant<__is_trivially_constructible(T, Args...)> {};
DPL_EXPORT template <typename T>
struct is_trivially_default_constructible :
    bool_constant<__is_trivially_constructible(T)> {};
DPL_EXPORT template <typename T>
struct is_trivially_copy_constructible :
    bool_constant<__is_trivially_constructible(
        T, add_lvalue_reference_t<T const>)> {};
DPL_EXPORT template <typename T>
struct is_trivially_move_constructible :
    bool_constant<__is_trivially_constructible(T, add_rvalue_reference_t<T>)> {
};

#else  // if __DPL_SHOULD_USE_BUILTIN(is_trivially_constructible)
DPL_EXPORT template <typename T, typename... Args>
inline constexpr bool is_trivially_constructible_v =
    unsupported_trait_v<T, Args...>;

DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_default_constructible_v =
    is_trivially_constructible_v<T>;

DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_copy_constructible_v =
    is_trivially_constructible_v<T, add_lvalue_reference_t<T const>>;
DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_move_constructible_v =
    is_trivially_constructible_v<T, add_rvalue_reference_t<T>>;

DPL_EXPORT template <typename T, typename... Args>
struct is_trivially_constructible :
    bool_constant<is_trivially_constructible_v<T, Args...>> {};
DPL_EXPORT template <typename T>
struct is_trivially_default_constructible :
    bool_constant<is_trivially_default_constructible_v<T>> {};
DPL_EXPORT template <typename T>
struct is_trivially_copy_constructible :
    bool_constant<is_trivially_copy_constructible_v<T>> {};
DPL_EXPORT template <typename T>
struct is_trivially_move_constructible :
    bool_constant<is_trivially_move_constructible_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_trivially_constructible)

DPL_DEFAULT_NAMESPACE_END
