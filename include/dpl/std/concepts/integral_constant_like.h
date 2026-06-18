// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/equality_comparable.h"
#include "dpl/std/concepts/integral.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/type_traits/remove_const.h"
#  include "dpl/std/type_traits/remove_cv.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details {
template <typename T>
concept integral_constant_like =
    requires { T::value; } && integral<decltype(T::value)> &&
    !same_as<bool, remove_const_t<decltype(T::value)>> &&
    convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> &&
    bool_constant<T() == T::value>::value &&
    bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value;
} // namespace details

DPL_EXPORT template <typename T>
concept integral_constant_like =
    details::integral_constant_like<remove_cv_t<T>>;

DPL_DEFAULT_NAMESPACE_END
