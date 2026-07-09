// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/std/type_traits/common_reference.h"
#  include "dpl/std/type_traits/is_assignable.h"
#  include "dpl/std/type_traits/is_constructible.h"
#  include "dpl/std/type_traits/is_convertible.h"
#  include "dpl/std/type_traits/is_integral.h"
#  include "dpl/std/type_traits/is_scalar.h"
#  include "dpl/std/type_traits/remove_const.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/type_traits/remove_reference.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT namespace details::concepts {

template <typename T, typename U>
concept oneway_same_as = is_same_v<T, U>;

template <typename T, typename U>
concept same_as = oneway_same_as<T, U> && oneway_same_as<U, T>;

template <typename T>
concept expensive_boolean_test = requires(T&& val, bool other) {
    requires !requires { operator&&(static_cast<T&&>(val), other); };
    requires !requires { operator&&(other, static_cast<T&&>(val)); };
    requires !requires {
        operator&&(static_cast<T&&>(val), static_cast<T&&>(val));
    };
    requires !requires { static_cast<T&&>(val).operator&&(other); };
    requires !requires {
        static_cast<T&&>(val).operator&&(static_cast<T&&>(val));
    };
    requires !requires { operator||(static_cast<T&&>(val), other); };
    requires !requires { operator||(other, static_cast<T&&>(val)); };
    requires !requires {
        operator||(static_cast<T&&>(val), static_cast<T&&>(val));
    };
    requires !requires { static_cast<T&&>(val).operator||(other); };
    requires !requires {
        static_cast<T&&>(val).operator||(static_cast<T&&>(val));
    };
    { other && static_cast<T&&>(val) } -> same_as<bool>;
    { static_cast<T&&>(val) && other } -> same_as<bool>;
    { other || static_cast<T&&>(val) } -> same_as<bool>;
    { static_cast<T&&>(val) || other } -> same_as<bool>;
};

template <typename From, typename To>
concept explicitly_convertible_to =
    requires(From&& val) { static_cast<To>(static_cast<From&&>(val)); };

template <typename From, typename To>
concept convertible_to =
    is_convertible_v<From, To> && explicitly_convertible_to<From, To>;

template <typename T>
concept boolean_testable = convertible_to<T, bool> &&
    (is_scalar_v<remove_cvref_t<T>> || expensive_boolean_test<T>);

template <typename T, typename U>
concept common_reference_with =
    same_as<common_reference_t<T, U>, common_reference_t<U, T>> &&
    convertible_to<T, common_reference_t<T, U>> &&
    convertible_to<U, common_reference_t<T, U>>;

template <typename T, typename U>
concept weakly_equality_comparable_with =
    requires(remove_reference_t<T> const& t, remove_reference_t<U> const& u) {
        { t == u } -> boolean_testable;
        { t != u } -> boolean_testable;
        { u == t } -> boolean_testable;
        { u != t } -> boolean_testable;
    };

template <typename T, typename U,
    typename C = common_reference_t<T const&, U const&>>
concept comparison_common_with_impl =
    same_as<common_reference_t<T const&, U const&>,
        common_reference_t<U const&, T const&>> &&
    requires {
        requires convertible_to<T const&, C const&> ||
            convertible_to<T, C const&>;
        requires convertible_to<U const&, C const&> ||
            convertible_to<U, C const&>;
    };

template <typename T, typename U>
concept comparison_common_with =
    common_reference_with<remove_cvref_t<T> const&, remove_cvref_t<U> const&> &&
    comparison_common_with_impl<remove_cvref_t<T>, remove_cvref_t<U>>;

template <typename T>
concept equality_comparable =
    details::concepts::weakly_equality_comparable_with<T, T>;

template <typename T, typename U>
concept equality_comparable_with = equality_comparable<T> &&
    equality_comparable<U> && comparison_common_with<T, U> &&
    equality_comparable<common_reference_t<remove_reference_t<T> const&,
        remove_reference_t<U> const&>> &&
    weakly_equality_comparable_with<T, U>;

template <typename T>
concept integral = is_integral_v<T>;

template <typename T>
concept integral_constant_like =
    requires { T::value; } && integral<decltype(T::value)> &&
    !same_as<bool, remove_const_t<decltype(T::value)>> &&
    convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> && (T() == T::value) &&
    (static_cast<decltype(T::value)>(T()) == T::value);

struct swap_t;

template <typename T>
void swap(T&, T&) = delete;

template <typename L, typename R>
concept unqualified_swappable =
    requires(L && (*l)(), R && (*r)()) { swap(l(), r()); };

template <typename L, typename R>
concept nothrow_unqualified_swappable =
    unqualified_swappable<L, R> && requires(L && (*l)(), R && (*r)()) {
        { swap(l(), r()) } noexcept;
    };

template <typename L, typename R, size_t N>
concept unqualified_swappable_array =
    (!unqualified_swappable<L (&)[N], R (&)[N]> && extent_v<L> == extent_v<R> &&
        requires(L (&l)[N], R (&r)[N], swap_t const func) { func(*l, *r); });

template <typename T>
concept exchangable = !unqualified_swappable<T&, T&> &&
    is_move_constructible_v<T> && is_move_assignable_v<T>;

struct swap_t {
    template <typename L, typename R>
    requires unqualified_swappable<L, R>
    __DPL_HIDE_FROM_ABI static constexpr void operator()(L&& l, R&& r) noexcept(
        nothrow_unqualified_swappable<L, R>) {
        swap(l, r);
    }

    template <typename L, typename R, size_t N>
    requires unqualified_swappable_array<L, R, N>
    __DPL_HIDE_FROM_ABI static constexpr void operator()(
        L (&l)[N], R (&r)[N]) noexcept(noexcept(swap_t::operator()(*l, *r))) {
        for (auto i = 0zu; i < N; ++i) {
            swap_t::operator()(l[i], r[i]);
        }
    }

    template <exchangable T>
    __DPL_HIDE_FROM_ABI static constexpr void operator()(T& l, T& r) noexcept(
        is_nothrow_move_constructible_v<T> && is_nothrow_move_assignable_v<T>) {
        T tmp(static_cast<T&&>(l));
        l = static_cast<T&&>(r);
        r = static_cast<T&&>(tmp);
    }
};

template <typename T, typename U>
concept partially_ordered_with =
    requires(remove_reference_t<T> const& t, remove_reference_t<U> const& u) {
        { t < u } -> boolean_testable;
        { t > u } -> boolean_testable;
        { t <= u } -> boolean_testable;
        { t >= u } -> boolean_testable;
        { u < t } -> boolean_testable;
        { u > t } -> boolean_testable;
        { u <= t } -> boolean_testable;
        { u >= t } -> boolean_testable;
    };
} // namespace details::concepts
DPL_DEFAULT_NAMESPACE_END
