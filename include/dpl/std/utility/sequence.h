// Copyright 2023-2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/stddef/types.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, T... Is>
struct integer_sequence {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto size() noexcept { return sizeof...(Is); }
};

#if DPL_HAS_BUILTIN(__make_integer_seq)

DPL_EXPORT template <typename T, T N>
using make_integer_sequence DPL_NODEBUG =
    __make_integer_seq<__DPL integer_sequence, T, N>;

#elif DPL_HAS_BUILTIN(__integer_pack)

DPL_EXPORT template <typename T, T N>
using make_integer_sequence DPL_NODEBUG =
    __DPL integer_sequence<T, __integer_pack(N)...>;

#else

namespace details {

template <typename, typename>
struct combine;

template <typename T, T... Is, T... Js>
struct combine<integer_sequence<T, Is...>, integer_sequence<T, Js...>> {
    using type DPL_NODEBUG =
        integer_sequence<T, Is..., (sizeof...(Is) + Js)...>;
};

template <typename T>
struct combine<integer_sequence<T>, integer_sequence<T>> {
    using type DPL_NODEBUG = integer_sequence<T>;
};

template <typename T, size_t N>
struct generate;

template <typename T>
struct generate<T, 0> {
    using type DPL_NODEBUG = integer_sequence<T>;
};

template <typename T>
struct generate<T, 1> {
    using type DPL_NODEBUG = integer_sequence<T, 0>;
};

template <typename T, size_t N>
struct generate :
    combine<typename generate<T, N / 2>::type,
        typename generate<T, N - N / 2>::type> {};

} // namespace details

DPL_EXPORT template <typename T, T N>
using make_integer_sequence DPL_NODEBUG = typename details::generate < T,
                            (N < 0) ? 0 : N > ::type;
#endif

DPL_EXPORT template <size_t... Is>
using index_sequence = integer_sequence<size_t, Is...>;

DPL_EXPORT template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

DPL_EXPORT template <typename... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

DPL_DEFAULT_NAMESPACE_END
