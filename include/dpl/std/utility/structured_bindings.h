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

DPL_EXPORT template <decltype(sizeof(0)), typename>
struct tuple_element {};

DPL_EXPORT template <typename T>
inline constexpr auto tuple_size_v = tuple_size<T>::value;

DPL_EXPORT template <decltype(sizeof(0)) N, typename T>
using tuple_element_t = typename tuple_element<N, T>::type;

DPL_DEFAULT_NAMESPACE_END

#if !DPL_IS_STD

DPL_STD_NAMESPACE_BEGIN

DPL_EXPORT template <typename>
struct tuple_size;

DPL_EXPORT template <decltype(sizeof(0)), typename>
struct tuple_element;

DPL_EXPORT template <typename T>
inline constexpr decltype(sizeof(0)) tuple_size_v = tuple_size<T>::value;

DPL_EXPORT template <decltype(sizeof(0)) N, typename T>
using tuple_element_t = typename tuple_element<N, T>::type;

DPL_STD_NAMESPACE_END

DPL_EXPORT template <typename T>
requires requires {
    typename __DPL integral_constant<size_t, __DPL tuple_size_v<T>>;
}
struct std::tuple_size<T> :
    __DPL integral_constant<size_t, __DPL tuple_size_v<T>> {};

DPL_EXPORT template <decltype(sizeof(0)) N, typename T>
requires requires { typename __DPL tuple_element_t<N, T>; }
struct std::tuple_element<N, T> : __DPL tuple_element<N, T> {};
#endif
