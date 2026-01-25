// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/is_void.h"
#if !__DPL_SHOULD_USE_BUILTIN(builtin_invoke)
#  include "dpl/std/type_traits/add_const.h"
#  include "dpl/std/type_traits/is_base_of.h"
#  include "dpl/std/type_traits/is_function.h"
#  include "dpl/std/type_traits/is_same.h"
#  include "dpl/std/type_traits/remove_const.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(builtin_invoke)

namespace details {
struct invoke_t {
    template <typename T>
    __DPL_HIDE_FROM_ABI static T&& decl(int) noexcept;
    template <typename T>
    __DPL_HIDE_FROM_ABI static T decl(float) noexcept;
    template <typename T>
    __DPL_HIDE_FROM_ABI static auto decl() noexcept
        -> decltype(invoke_t::decl<T>(0));
};
} // namespace details

#  define __DPL_BUILTIN_invoke(...) __builtin_invoke(__VA_ARGS__)

#else

namespace details {
template <typename>
inline constexpr bool is_wrapped = false;
__DPL_HIDE_FROM_ABI void ref(...) noexcept = delete;
__DPL_HIDE_FROM_ABI void cref(...) noexcept = delete;

template <typename T, typename U>
concept reference_of = is_same_v<U&, T> && is_same_v<T, U&>;
template <typename T, typename U>
concept same_result = is_same_v<U, T> && is_same_v<T, U>;
template <template <typename> class Wrapper, typename T>
concept wrapped_impl = requires(Wrapper<T>* wrap) {
    wrap->~reference_wrapper();
    { static_cast<Wrapper<T> const*>(wrap)->get() } noexcept -> reference_of<T>;
    { wrap->get() } noexcept -> reference_of<T>;
    static_cast<void (*)(T&)>(0)(*static_cast<Wrapper<T> const*>(wrap));
    static_cast<void (*)(T&)>(0)(*wrap);
    // adl would pick these up
    { ref(*wrap) } noexcept -> same_result<Wrapper<T>>;
    { cref(*wrap) } noexcept -> same_result<Wrapper<add_const_t<T>>>;
};

template <template <typename> class Wrapper, typename T>
requires reference_of<T&, T&>
inline constexpr bool is_wrapped<Wrapper<T>> =
    wrapped_impl<Wrapper, remove_const_t<T>> &&
    wrapped_impl<Wrapper, add_const_t<T>>;

template <typename T>
concept object = is_object_v<T>;

template <typename T>
concept function = is_function_v<T>;

template <typename T>
concept not_function = !function<T>;

template <typename O, typename C>
concept not_reference_of = object<remove_cvref_t<O>> &&
    !is_same_v<C, remove_cvref_t<O>> && !is_base_of_v<C, remove_cvref_t<O>>;

template <typename T>
concept not_ref_wrapper = !is_wrapped<remove_cvref_t<T>>;

struct invoke_t {
    template <typename T>
    __DPL_HIDE_FROM_ABI static T&& decl(int) noexcept;
    template <typename T>
    __DPL_HIDE_FROM_ABI static T decl(float) noexcept;
    template <typename T>
    __DPL_HIDE_FROM_ABI static auto decl() noexcept
        -> decltype(invoke_t::decl<T>(0));

    DPL_EXPORT template <typename F, typename... Args>
    __DPL_HIDE_FROM_ABI constexpr auto
    operator()(F&& func, Args&&... args) noexcept(
        noexcept(invoke_t::decl<F>()(invoke_t::decl<Args>()...)))
        -> decltype(invoke_t::decl<F>()(invoke_t::decl<Args>()...)) {
        return static_cast<F&&>(func)(static_cast<Args&&>(args)...);
    }

    template <object C, not_function P, object O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(P C::* member, O&& obj) noexcept
        -> decltype(invoke_t::decl<O>().*member) {
        return static_cast<O&&>(obj).*member;
    }

    template <object C, not_function P, not_reference_of<C> O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(P C::* member, O&& obj) noexcept
        -> decltype(obj.get().*member) {
        return obj.get().*member;
    }

    template <object C, not_function P, not_reference_of<C> O>
    requires not_ref_wrapper<O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(P C::* member, O&& obj) noexcept
        -> decltype((*invoke_t::decl<O>()).*member) {
        return (*static_cast<O&&>(obj)).*member;
    }

    template <object C, function P, object O, typename... Args>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(
        P C::* member, O&& obj, Args&&... args) noexcept
        -> decltype((invoke_t::decl<O>().*member)(invoke_t::decl<Args>()...)) {
        return (static_cast<O&&>(obj).*member)(static_cast<Args&&>(args)...);
    }

    template <object C, function P, not_reference_of<C> O, typename... Args>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(
        P C::* member, O&& obj, Args&&... args) noexcept
        -> decltype((obj.get().*member)(invoke_t::decl<Args>()...)) {
        return (obj.get().*member)(static_cast<Args&&>(args)...);
    }

    template <object C, function P, not_reference_of<C> O, typename... Args>
    requires not_ref_wrapper<O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(P C::* member, O&& obj,
        Args&&... args) noexcept -> decltype(((*invoke_t::decl<O>()).*
        member)(invoke_t::decl<Args>()...)) {
        return ((*static_cast<O&&>(obj)).*member)(static_cast<Args&&>(args)...);
    }
};

inline constexpr invoke_t invoke{};

} // namespace details

#  define __DPL_BUILTIN_invoke(...) __DPL details::invoke(__VA_ARGS__)

#endif // if __DPL_SHOULD_USE_BUILTIN(is_invocable)

DPL_EXPORT template <typename F, typename... Args>
using invoke_result_t = decltype(__DPL_BUILTIN_invoke(
    details::invoke_t::decl<F>(), details::invoke_t::decl<Args>()...));

DPL_EXPORT template <typename F, typename... Args>
struct invoke_result {};

DPL_EXPORT template <typename F, typename... Args>
requires requires { typename invoke_result_t<F, Args...>; }
struct invoke_result<F, Args...> {
    using type DPL_NODEBUG = invoke_result_t<F, Args...>;
};

DPL_EXPORT template <typename F, typename... Args>
inline constexpr bool is_invocable_v =
    requires { typename invoke_result_t<F, Args...>; };

DPL_EXPORT template <typename F, typename... Args>
inline constexpr bool is_nothrow_invocable_v = false;

DPL_EXPORT template <typename F, typename... Args>
requires is_invocable_v<F, Args...>
inline constexpr bool is_nothrow_invocable_v<F, Args...> =
    noexcept(__DPL_BUILTIN_invoke(
        details::invoke_t::decl<F>(), details::invoke_t::decl<Args>()...));

DPL_EXPORT template <typename R, typename F, typename... Args>
inline constexpr bool is_invocable_r_v = false;

DPL_EXPORT template <typename R, typename F, typename... Args>
requires is_invocable_v<F, Args...> &&
    (is_void_v<R> || is_void_v<invoke_result_t<F, Args...>>)
inline constexpr bool is_invocable_r_v<R, F, Args...> =
    !is_void_v<invoke_result_t<F, Args...>> || is_void_v<R>;

DPL_EXPORT template <typename R, typename F, typename... Args>
requires is_invocable_v<F, Args...>
inline constexpr bool is_invocable_r_v<R, F, Args...> =
    requires { [](invoke_result_t<F, Args...> arg) -> R { return arg; }; };

DPL_EXPORT template <typename R, typename F, typename... Args>
inline constexpr bool is_nothrow_invocable_r_v = false;

DPL_EXPORT template <typename R, typename F, typename... Args>
requires is_nothrow_invocable_v<F, Args...> &&
    is_invocable_r_v<R, F, Args...> &&
    (is_void_v<R> || is_void_v<invoke_result_t<F, Args...>>)
inline constexpr bool is_nothrow_invocable_r_v<R, F, Args...> = true;

DPL_EXPORT template <typename R, typename F, typename... Args>
requires is_nothrow_invocable_v<F, Args...> && is_invocable_r_v<R, F, Args...>
inline constexpr bool is_nothrow_invocable_r_v<R, F, Args...> =
    requires(invoke_result_t<F, Args...> result) {
        { static_cast<R>(result) } noexcept;
    };

DPL_EXPORT template <typename F, typename... Args>
struct is_invocable : bool_constant<is_invocable_v<F, Args...>> {};
DPL_EXPORT template <typename R, typename F, typename... Args>
struct is_invocable_r : bool_constant<is_invocable_r_v<R, F, Args...>> {};
DPL_EXPORT template <typename F, typename... Args>
struct is_nothrow_invocable :
    bool_constant<is_nothrow_invocable_v<F, Args...>> {};
DPL_EXPORT template <typename R, typename F, typename... Args>
struct is_nothrow_invocable_r :
    bool_constant<is_nothrow_invocable_r_v<R, F, Args...>> {};

DPL_EXPORT template <typename F, typename... Args>
__DPL_HIDE_FROM_ABI constexpr invoke_result_t<F, Args...> invoke(
    F&& func, Args&&... args) noexcept(is_nothrow_invocable_v<F, Args...>) {
    return __DPL_BUILTIN_invoke(
        static_cast<F&&>(func)(static_cast<Args&&>(args)...));
}

DPL_EXPORT template <typename R, typename F, typename... Args>
requires is_invocable_r_v<R, F, Args...>
__DPL_HIDE_FROM_ABI constexpr R invoke_r(F&& func, Args&&... args) noexcept(
    is_nothrow_invocable_r_v<R, F, Args...>) {
    if constexpr (is_void_v<R>) {
        __DPL_BUILTIN_invoke(
            static_cast<F&&>(func)(static_cast<Args&&>(args)...));
    } else {
        return __DPL_BUILTIN_invoke(
            static_cast<F&&>(func)(static_cast<Args&&>(args)...));
    }
}

#undef __DPL_BUILTIN_invoke

DPL_DEFAULT_NAMESPACE_END
