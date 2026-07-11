// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/assignable_from.h"
#include "dpl/std/concepts/constructible_from.h"
#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/move_constructible.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/is_nothrow_assignable.h"
#  include "dpl/std/type_traits/is_nothrow_constructible.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::concepts {

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
concept exchangable = !unqualified_swappable<T&, T&> && move_constructible<T> &&
    assignable_from<T&, T>;

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
        for (decltype(N) i = 0; i < N; ++i) {
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

} // namespace details::concepts

namespace ranges {
inline namespace cpo {
inline constexpr __DPL details::concepts::swap_t swap{};
}
} // namespace ranges

template <typename T, typename U>
concept swappable_with =
    common_reference_with<T, U> && requires(T && (*lhs)(), U && (*rhs)()) {
        ranges::swap(lhs(), lhs());
        ranges::swap(rhs(), rhs());
        ranges::swap(lhs(), rhs());
        ranges::swap(rhs(), lhs());
    };

template <typename T>
concept swappable = requires(T& lhs, T& rhs) { ranges::swap(lhs, rhs); };

__DPL_DEFAULT_NAMESPACE_END
