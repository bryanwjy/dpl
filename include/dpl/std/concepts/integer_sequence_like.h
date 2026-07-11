// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/integral.h"
#include "dpl/std/concepts/integral_constant_like.h"
#include "dpl/std/concepts/semiregular.h"
#include "dpl/std/concepts/tuple_like.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/type_traits/remove_cv.h"
#  include "dpl/std/type_traits/structured_bindings.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace details::concepts {

template <typename T, typename E>
concept integer_sequence_like = integral<E> && semiregular<T> &&
    requires { typename size_constant<T::size()>; } && tuple_like<T> &&
    (T::size() == std::tuple_size_v<T>) &&
    []<size_t I>(this auto self, size_constant<I>) {
        if constexpr (I == T::size()) {
            return true;
        } else {
            return integral_constant_like<std::tuple_element_t<I, T>> &&
                convertible_to<std::tuple_element_t<I, T>, E> &&
                self(size_constant<I + 1>{});
        }
    }(size_constant<0zu>{});

template <typename E, typename T, size_t I, E... Vs>
consteval auto to_integer_sequence(
    size_constant<I>, integral_constant<E, Vs>... args) noexcept {
    if constexpr (I == T::size()) {
        return integer_sequence<E, Vs...>{};
    } else {
        constexpr auto V = ranges::get_element<I>(T{});
        constexpr integral_constant<E, V> next{};
        return to_integer_sequence<E, T>(size_constant<I + 1>{}, args..., next);
    }
}

template <typename E, typename T>
inline constexpr bool is_integer_sequence = false;

template <typename E, E... Vs>
inline constexpr bool is_integer_sequence<E, integer_sequence<E, Vs...>> = true;

template <typename T, typename E>
concept integer_sequence_of = is_integer_sequence<E, T>;

template <integral E, details::concepts::integer_sequence_of<E> T>
consteval auto to_integer_sequence(T seq = T{}) noexcept {
    return seq;
}

template <integral E, typename T>
requires (!details::concepts::integer_sequence_of<T, E> &&
    details::concepts::integer_sequence_like<T, E>)
consteval auto to_integer_sequence(T = T{}) noexcept {
    if constexpr (T::size() == 0) {
        return integer_sequence<E>{};
    } else {
        return details::concepts::to_integer_sequence<E, T>(
            size_constant<0zu>{});
    }
}

} // namespace details::concepts

template <typename T, typename E>
concept integer_sequence_like =
    details::concepts::integer_sequence_like<remove_cv_t<T>, E> && requires {
        {
            details::concepts::to_integer_sequence<E, remove_cv_t<T>>()
        } -> details::concepts::integer_sequence_of<E>;
    };

template <integral E, E... Is>
consteval auto to_integer_sequence(integer_sequence<E, Is...> seq) noexcept {
    return seq;
}

template <integral E, integer_sequence_like<E> T>
consteval auto to_integer_sequence(T seq = T{}) noexcept {
    return details::concepts::to_integer_sequence<E>(seq);
}

template <typename T>
concept index_sequence_like = integer_sequence_like<T, size_t>;

template <size_t... Is>
consteval auto to_index_sequence(index_sequence<Is...> seq) noexcept {
    return seq;
}

template <index_sequence_like T>
consteval auto to_index_sequence(T seq = T{}) noexcept {
    return details::concepts::to_integer_sequence<size_t>(seq);
}

__DPL_DEFAULT_NAMESPACE_END
