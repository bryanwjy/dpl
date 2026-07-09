// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/std/type_traits/constants.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename>
struct tuple_size {};

DPL_EXPORT template <typename T>
struct tuple_size<T const> : tuple_size<T> {};
DPL_EXPORT template <typename T>
struct tuple_size<T volatile> : tuple_size<T> {};
DPL_EXPORT template <typename T>
struct tuple_size<T const volatile> : tuple_size<T> {};

DPL_EXPORT template <decltype(sizeof(0)), typename>
struct tuple_element {};

DPL_EXPORT template <decltype(sizeof(0)) I, typename T>
struct tuple_element<I, T const> : tuple_element<I, T> {};
DPL_EXPORT template <decltype(sizeof(0)) I, typename T>
struct tuple_element<I, T volatile> : tuple_element<I, T> {};
DPL_EXPORT template <decltype(sizeof(0)) I, typename T>
struct tuple_element<I, T const volatile> : tuple_element<I, T> {};

DPL_EXPORT template <typename T>
inline constexpr auto tuple_size_v = tuple_size<T>::value;

DPL_EXPORT template <decltype(sizeof(0)) N, typename T>
using tuple_element_t = typename tuple_element<N, T>::type;

DPL_DEFAULT_NAMESPACE_END

#if !DPL_IS_STD

DPL_EXPORT template <typename T>
requires requires {
    __DPL tuple_size<T>::value;
    typename __DPL integral_constant<size_t, __DPL tuple_size_v<T>>;
}
struct std::tuple_size<T> :
    __DPL integral_constant<size_t, __DPL tuple_size_v<T>> {};

DPL_EXPORT template <decltype(sizeof(0)) N, typename T>
requires requires { typename __DPL tuple_element_t<N, T>; }
struct std::tuple_element<N, T> : __DPL tuple_element<N, T> {};
#endif
