// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/add_lvalue_reference.h"
#include "dpl/std/type_traits/add_rvalue_reference.h"
#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_assignable)
template <typename T, typename R>
inline constexpr bool is_assignable_v = __is_assignable(T, R);
template <typename T>
inline constexpr bool is_copy_assignable_v =
    __is_assignable(add_lvalue_reference_t<T>, add_lvalue_reference_t<T const>);
template <typename T>
inline constexpr bool is_move_assignable_v =
    __is_assignable(add_lvalue_reference_t<T>, add_rvalue_reference_t<T>);

template <typename T, typename... Args>
struct is_assignable : bool_constant<__is_assignable(T, Args...)> {};
template <typename T>
struct is_copy_assignable :
    bool_constant<__is_assignable(
        add_lvalue_reference_t<T>, add_lvalue_reference_t<T const>)> {};
template <typename T>
struct is_move_assignable :
    bool_constant<__is_assignable(
        add_lvalue_reference_t<T>, add_rvalue_reference_t<T>)> {};

#else  // if __DPL_SHOULD_USE_BUILTIN(is_assignable)
template <typename T, typename R>
inline constexpr bool is_assignable_v =
    requires(R&& arg) { T(static_cast<R&&>(arg)); };

template <typename T>
inline constexpr bool is_copy_assignable_v =
    is_assignable_v<add_lvalue_reference_t<T>, add_lvalue_reference_t<T const>>;
template <typename T>
inline constexpr bool is_move_assignable_v =
    is_assignable_v<add_lvalue_reference_t<T>, add_rvalue_reference_t<T>>;

template <typename T, typename R>
struct is_assignable : bool_constant<is_assignable_v<T, R>> {};
template <typename T>
struct is_copy_assignable : bool_constant<is_copy_assignable_v<T>> {};
template <typename T>
struct is_move_assignable : bool_constant<is_move_assignable_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_assignable)

__DPL_DEFAULT_NAMESPACE_END
