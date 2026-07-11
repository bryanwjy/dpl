// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/std/type_traits/constants.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename>
struct tuple_size {};

template <typename T>
struct tuple_size<T const> : tuple_size<T> {};
template <typename T>
struct tuple_size<T volatile> : tuple_size<T> {};
template <typename T>
struct tuple_size<T const volatile> : tuple_size<T> {};

template <decltype(sizeof(0)), typename>
struct tuple_element {};

template <decltype(sizeof(0)) I, typename T>
struct tuple_element<I, T const> : tuple_element<I, T> {};
template <decltype(sizeof(0)) I, typename T>
struct tuple_element<I, T volatile> : tuple_element<I, T> {};
template <decltype(sizeof(0)) I, typename T>
struct tuple_element<I, T const volatile> : tuple_element<I, T> {};

template <typename T>
inline constexpr auto tuple_size_v = tuple_size<T>::value;

template <decltype(sizeof(0)) N, typename T>
using tuple_element_t = typename tuple_element<N, T>::type;

__DPL_DEFAULT_NAMESPACE_END

#if !DPL_IS_STD

template <typename T>
requires requires {
    __DPL tuple_size<T>::value;
    typename __DPL integral_constant<__DPL size_t, __DPL tuple_size_v<T>>;
}
struct std::tuple_size<T> :
    __DPL integral_constant<__DPL size_t, __DPL tuple_size_v<T>> {};

template <decltype(sizeof(0)) N, typename T>
requires requires { typename __DPL tuple_element_t<N, T>; }
struct std::tuple_element<N, T> : __DPL tuple_element<N, T> {};
#endif
