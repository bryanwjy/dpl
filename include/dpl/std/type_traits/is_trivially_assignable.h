// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/add_lvalue_reference.h"
#include "dpl/std/type_traits/add_rvalue_reference.h"
#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_trivially_assignable)
DPL_EXPORT template <typename T, typename... Args>
inline constexpr bool is_trivially_assignable_v =
    __is_trivially_assignable(T, Args...);
DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_copy_assignable_v =
    __is_trivially_assignable(T, add_lvalue_reference_t<T const>);
DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_move_assignable_v =
    __is_trivially_assignable(T, add_rvalue_reference_t<T>);

DPL_EXPORT template <typename T, typename... Args>
struct is_trivially_assignable :
    bool_constant<__is_trivially_assignable(T, Args...)> {};
DPL_EXPORT template <typename T>
struct is_trivially_copy_assignable :
    bool_constant<__is_trivially_assignable(
        T, add_lvalue_reference_t<T const>)> {};
DPL_EXPORT template <typename T>
struct is_trivially_move_assignable :
    bool_constant<__is_trivially_assignable(T, add_rvalue_reference_t<T>)> {};

#else  // if __DPL_SHOULD_USE_BUILTIN(is_trivially_assignable)
DPL_EXPORT template <typename T, typename... Args>
inline constexpr bool is_trivially_assignable_v =
    requires(Args&&... args) { T(static_cast<Args&&>(args)...); };

DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_copy_assignable_v =
    is_trivially_assignable_v<T, add_lvalue_reference_t<T const>>;
DPL_EXPORT template <typename T>
inline constexpr bool is_trivially_move_assignable_v =
    is_trivially_assignable_v<T, add_rvalue_reference_t<T>>;

DPL_EXPORT template <typename T, typename... Args>
struct is_trivially_assignable :
    bool_constant<is_trivially_assignable_v<T, Args...>> {};
DPL_EXPORT template <typename T>
struct is_trivially_copy_assignable :
    bool_constant<is_trivially_copy_assignable_v<T>> {};
DPL_EXPORT template <typename T>
struct is_trivially_move_assignable :
    bool_constant<is_trivially_move_assignable_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_trivially_assignable)

DPL_DEFAULT_NAMESPACE_END
