// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/same_as.h"
#if !DPL_MODULES
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::concepts {
template <typename T>
concept expensive_boolean_test = requires(T&& val, bool other) {
    requires !requires { operator&&(static_cast<T&&>(val), other); };
    requires !requires { operator&&(other, static_cast<T&&>(val)); };
    requires !requires {
        operator&&(static_cast<T&&>(val), static_cast<T&&>(val));
    };
    requires !requires { static_cast<T&&>(val).operator&&(other); };
    requires !requires {
        static_cast<T&&>(val).operator&&(static_cast<T&&>(val));
    };
    requires !requires { operator||(static_cast<T&&>(val), other); };
    requires !requires { operator||(other, static_cast<T&&>(val)); };
    requires !requires {
        operator||(static_cast<T&&>(val), static_cast<T&&>(val));
    };
    requires !requires { static_cast<T&&>(val).operator||(other); };
    requires !requires {
        static_cast<T&&>(val).operator||(static_cast<T&&>(val));
    };
    { other && static_cast<T&&>(val) } -> same_as<bool>;
    { static_cast<T&&>(val) && other } -> same_as<bool>;
    { other || static_cast<T&&>(val) } -> same_as<bool>;
    { static_cast<T&&>(val) || other } -> same_as<bool>;
};

template <typename T>
concept boolean_testable = convertible_to<T, bool> &&
    (is_scalar_v<remove_cvref_t<T>> || expensive_boolean_test<T>);

} // namespace details::concepts

DPL_EXPORT template <typename T>
concept boolean_testable =
    details::concepts::boolean_testable<T> && requires(T&& val) {
        { !static_cast<T&&>(val) } -> details::concepts::boolean_testable;
    };

DPL_DEFAULT_NAMESPACE_END
