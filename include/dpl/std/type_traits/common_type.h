// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/decay.h"
#include "dpl/std/type_traits/declval.h"
#include "dpl/std/type_traits/is_same.h"
#include "dpl/std/type_traits/remove_reference.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename...>
struct common_type {};
DPL_EXPORT template <>
struct common_type<> {};
DPL_EXPORT template <typename T>
struct common_type<T> {
    using type = T;
};

DPL_EXPORT template <typename... Ts>
using common_type_t DPL_NODEBUG = typename common_type<Ts...>::type;

namespace details::common_type {
template <typename T, typename U>
using ternary_result_t DPL_NODEBUG = decltype([]() {
    return false ? __DPL declval<T>() : __DPL declval<U>();
}());

template <typename T, typename U>
struct impl2 {};
template <typename T, typename U>
struct impl1 : impl2<T, U> {};
template <typename T, typename U>
struct impl0 : impl1<T, U> {};

template <typename T, typename U>
requires (!is_same_v<T, decay_t<T>> || !is_same_v<U, decay_t<U>>)
struct impl0<T, U> : __DPL common_type<decay_t<T>, decay_t<U>> {};

template <typename T, typename U>
requires requires { typename ternary_result_t<T, U>; }
struct impl1<T, U> {
    using type DPL_NODEBUG = ternary_result_t<T, U>;
};

template <typename T, typename U>
requires requires {
    typename ternary_result_t<remove_reference_t<T> const&,
        remove_reference_t<U> const&>;
}
struct impl2<T, U> {
    using type DPL_NODEBUG = ternary_result_t<remove_reference_t<T> const&,
        remove_reference_t<U> const&>;
};

template <typename T, typename U>
struct impl : impl0<T, U> {};

template <typename T, typename U>
requires requires { typename __DPL common_type_t<T, U>; }
struct impl<T, U> {
    using type DPL_NODEBUG = __DPL common_type_t<T, U>;
};

} // namespace details::common_type

DPL_EXPORT template <typename T, typename U>
struct common_type<T, U> : details::common_type::impl<T, U> {};

DPL_EXPORT template <typename T, typename U, typename... Vs>
requires requires { typename details::common_type::impl<T, U>::type; }
struct common_type<T, U, Vs...> : common_type<common_type_t<T, U>, Vs...> {};

DPL_DEFAULT_NAMESPACE_END
