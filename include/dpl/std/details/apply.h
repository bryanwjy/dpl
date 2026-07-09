// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/tuple_like.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/type_traits/structured_bindings.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT namespace details::apply {
template <typename T>
struct safe_sequence {
    using type DPL_NODEBUG = void;
};

template <tuple_like T>
struct safe_sequence<T> {
    using type DPL_NODEBUG = __DPL make_index_sequence<
        std::tuple_size_v<__DPL remove_cvref_t<T>>>;
};

template <size_t I, typename T>
requires requires { ranges::get_element<I>(__DPL declval<T>()); }
using decl_element_t DPL_NODEBUG =
    decltype(ranges::get_element<I>(__DPL declval<T>()));

template <typename F, typename T, typename = typename safe_sequence<T>::type>
inline constexpr bool is_applicable_v = false;

template <typename F, tuple_like T, size_t... Is>
inline constexpr bool is_applicable_v<F, T, index_sequence<Is...>> =
    __DPL is_invocable_v<F, decl_element_t<Is, T>...>;

template <typename F, typename T, typename = typename safe_sequence<T>::type>
inline constexpr bool is_nothrow_applicable_v = false;

template <typename F, tuple_like T, size_t... Is>
requires is_applicable_v<F, T>
inline constexpr bool is_nothrow_applicable_v<F, T, index_sequence<Is...>> =
    __DPL is_nothrow_invocable_v<F, decl_element_t<Is, T>...> &&
    (... && noexcept(ranges::get_element<Is>(__DPL declval<T>())));
} // namespace details::apply

DPL_DEFAULT_NAMESPACE_END
