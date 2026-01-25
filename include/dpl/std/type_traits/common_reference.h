// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/add_const.h"
#include "dpl/std/type_traits/add_pointer.h"
#include "dpl/std/type_traits/add_rvalue_reference.h"
#include "dpl/std/type_traits/add_volatile.h"
#include "dpl/std/type_traits/common_type.h"
#include "dpl/std/type_traits/conditional.h"
#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/copy_cvref.h"
#include "dpl/std/type_traits/is_const.h"
#include "dpl/std/type_traits/is_convertible.h"
#include "dpl/std/type_traits/is_same.h"
#include "dpl/std/type_traits/is_volatile.h"
#include "dpl/std/type_traits/remove_cvref.h"
#include "dpl/std/type_traits/remove_reference.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, typename U, template <typename> class TQual,
    template <typename> class UQual>
struct basic_common_reference {};
DPL_EXPORT template <typename T, typename U, template <typename> class TQual,
    template <typename> class UQual>
using basic_common_reference_t =
    typename basic_common_reference<T, U, TQual, UQual>::type;
DPL_EXPORT template <typename...>
struct common_reference {};
DPL_EXPORT template <typename... Ts>
using common_reference_t = typename common_reference<Ts...>::type;
DPL_EXPORT template <>
struct common_reference<> {};
DPL_EXPORT template <typename T>
struct common_reference<T> {
    using type = T;
};

namespace details::common_reference {

template <typename T>
struct copy_cvref_from {
    template <typename U>
    using apply = copy_cvref_t<T, U>;
};

template <typename T0, typename T1>
struct ternary_result {};

template <typename T0, typename T1>
requires requires {
    false ? static_cast<T0 (*)()>(0)() : static_cast<T1 (*)()>(0)();
}
struct ternary_result<T0, T1> {
    using type = decltype(false ? static_cast<T0 (*)()>(0)()
                                : static_cast<T1 (*)()>(0)());
};

template <typename T0, typename T1>
using ternary_result_t = typename ternary_result<T0, T1>::type;

template <typename Target, typename... Ts>
using merge_const_t =
    conditional_t<(... || is_const_v<Ts>), add_const_t<Target>, Target>;
template <typename Target, typename... Ts>
using merge_volatile_t =
    conditional_t<(... || is_volatile_v<Ts>), add_volatile_t<Target>, Target>;

template <typename Target, typename... Ts>
using merge_cv_t = merge_volatile_t<merge_const_t<Target, Ts...>, Ts...>;

template <typename T0, typename T1>
struct simple_common_ref {};

template <typename T0, typename T1>
using simple_common_ref_t = typename simple_common_ref<T0, T1>::type;

template <typename Target, typename... Ts>
concept all_convertible_from = (... && is_convertible_v<Ts, Target>);

template <typename T0, typename T1>
requires all_convertible_from<
    add_pointer_t<
        ternary_result_t<merge_cv_t<T0, T0, T1>&, merge_cv_t<T1, T0, T1>&> //
        >,
    add_pointer_t<T0&>, //
    add_pointer_t<T1&>>
struct simple_common_ref<T0&, T1&> {
    using type =
        ternary_result_t<merge_cv_t<T0, T0, T1>&, merge_cv_t<T1, T0, T1>&>;
};

template <typename T>
using rvalue_ref = add_rvalue_reference<remove_reference_t<T>>;
template <typename T>
using rvalue_ref_t = typename rvalue_ref<T>::type;

template <typename T0, typename T1>
requires all_convertible_from<rvalue_ref_t<simple_common_ref_t<T0&, T1&>>, T0&&,
             T1&&> &&
    all_convertible_from<
        add_pointer_t<rvalue_ref_t<simple_common_ref_t<T0&, T1&>>>,
        add_pointer_t<T0&&>, add_pointer_t<T1&&>>
struct simple_common_ref<T0&&, T1&&> :
    rvalue_ref<simple_common_ref_t<T0&, T1&>> {};

template <typename T0, typename T1>
requires is_convertible_v<T1&&, simple_common_ref_t<T0&, add_const_t<T1>&>>
struct simple_common_ref<T0&, T1&&> :
    simple_common_ref<T0&, add_const_t<T1>&> {};

template <typename T0, typename T1>
requires requires { typename simple_common_ref_t<T1&, T0&&>; }
struct simple_common_ref<T0&&, T1&> : simple_common_ref<T1&, T0&&> {};

template <typename T, typename U>
using basic_common_ref =
    basic_common_reference<remove_cvref_t<T>, remove_cvref_t<U>,
        copy_cvref_from<T>::template apply, copy_cvref_from<U>::template apply>;

template <typename...>
struct first_type;

template <>
struct first_type<> {};

template <typename T, typename... Ts>
requires requires { typename T::type; }
struct first_type<T, Ts...> : T {};

template <typename T, typename... Ts>
struct first_type<T, Ts...> : first_type<Ts...> {};

template <typename T, typename U>
using impl = first_type<simple_common_ref<T, U>,
    basic_common_reference<remove_cvref_t<T>, remove_cvref_t<U>,
        copy_cvref_from<T>::template apply, copy_cvref_from<U>::template apply>,
    ternary_result<T, U>, __DPL common_type<T, U>>;

} // namespace details::common_reference

DPL_EXPORT template <typename T, typename U>
struct common_reference<T, U> : details::common_reference::impl<T, U> {};

DPL_EXPORT template <typename T, typename U, typename... Vs>
requires requires { typename details::common_reference::impl<T, U>::type; }
struct common_reference<T, U, Vs...> :
    common_reference<common_reference_t<T, U>, Vs...> {};

DPL_DEFAULT_NAMESPACE_END
