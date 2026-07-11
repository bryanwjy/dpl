// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/decay.h"
#include "dpl/std/type_traits/declval.h"
#include "dpl/std/type_traits/is_same.h"
#include "dpl/std/type_traits/remove_reference.h"
#include "dpl/std/type_traits/ternary_result.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename...>
struct common_type {};
template <>
struct common_type<> {};
template <typename T>
struct common_type<T> {
    using type = T;
};
template <typename T>
struct common_type<T, T> {
    using type = T;
};

template <typename... Ts>
using common_type_t DPL_NODEBUG = typename common_type<Ts...>::type;

namespace details::type_traits {

template <typename T, typename U>
struct common_type2 {};
template <typename T, typename U>
struct common_type1 : common_type2<T, U> {};
template <typename T, typename U>
struct common_type0 : common_type1<T, U> {};

template <typename T, typename U>
requires (!is_same_v<T, decay_t<T>> || !is_same_v<U, decay_t<U>>)
struct common_type0<T, U> : __DPL common_type<decay_t<T>, decay_t<U>> {};

template <typename T, typename U>
requires requires { typename ternary_result_t<T, U>; }
struct common_type1<T, U> {
    using type DPL_NODEBUG = ternary_result_t<T, U>;
};

template <typename T, typename U>
requires requires {
    typename ternary_result_t<remove_reference_t<T> const&,
        remove_reference_t<U> const&>;
}
struct common_type2<T, U> {
    using type DPL_NODEBUG = ternary_result_t<remove_reference_t<T> const&,
        remove_reference_t<U> const&>;
};

template <typename T, typename U>
struct common_type : common_type0<T, U> {};

} // namespace details::type_traits

template <typename T, typename U>
struct common_type<T, U> : details::type_traits::common_type<T, U> {};

template <typename T, typename U, typename... Vs>
requires requires { typename common_type<T, U>::type; }
struct common_type<T, U, Vs...> : common_type<common_type_t<T, U>, Vs...> {};

__DPL_DEFAULT_NAMESPACE_END
