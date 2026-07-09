// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/integral.h"
#include "dpl/std/concepts/semiregular.h"

#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#  include "dpl/std/type_traits/remove_cv.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, typename E>
concept integer_sequence_like =
    details::concepts::integer_sequence_like<remove_cv_t<T>, E> &&
    semiregular<T> && integral<E> && requires {
        {
            details::concepts::to_integer_sequence<E, remove_cv_t<T>>()
        } -> details::concepts::integer_sequence_of<E>;
    };

DPL_EXPORT template <integral E, E... Is>
consteval auto to_integer_sequence(integer_sequence<E, Is...> seq) noexcept {
    return seq;
}

DPL_EXPORT template <integral E, integer_sequence_like<E> T>
consteval auto to_integer_sequence(T seq = T{}) noexcept {
    return details::concepts::to_integer_sequence<E>(seq);
}

DPL_EXPORT template <typename T>
concept index_sequence_like = integer_sequence_like<T, size_t>;

DPL_EXPORT template <size_t... Is>
consteval auto to_index_sequence(index_sequence<Is...> seq) noexcept {
    return seq;
}

DPL_EXPORT template <index_sequence_like T>
consteval auto to_index_sequence(T seq = T{}) noexcept {
    return details::concepts::to_integer_sequence<size_t>(seq);
}

DPL_DEFAULT_NAMESPACE_END
