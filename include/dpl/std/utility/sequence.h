// Copyright 2023-2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/structured_bindings.h"

#if !DPL_MODULES
#  include "dpl/std/stddef/types.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, T... Is>
struct integer_sequence {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto size() noexcept { return sizeof...(Is); }
};

DPL_EXPORT template <typename T, T... Is>
struct tuple_size<integer_sequence<T, Is...>> : size_constant<sizeof...(Is)> {};

DPL_EXPORT template <size_t I, typename T, T... Is>
requires (I < sizeof...(Is))
struct tuple_element<I, integer_sequence<T, Is...>> {
#if __cpp_pack_indexing >= 202311L && DPL_HAS_CXX26_EXTENSIONS
    using type = integral_constant<T, Is...[I]>;
#else
private:
    static consteval auto get_type() noexcept {
        using array = T[sizeof...(Is)];
        constexpr array vals{Is...};
        return integral_constant<T, vals[I]>{};
    }

public:
    using type = decltype(get_type());
#endif
};

template <size_t I, typename T, T... Is>
consteval auto get(integer_sequence<T, Is...>) noexcept {
    return tuple_element_t<I, integer_sequence<T, Is...>>{};
}

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

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
