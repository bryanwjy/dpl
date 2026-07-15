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

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::concepts {
template <typename T>
concept integral_constant_like =
    requires { T::value; } && integral<decltype(T::value)> &&
    !same_as<bool, remove_const_t<decltype(T::value)>> &&
    convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> && (T() == T::value) &&
    (static_cast<decltype(T::value)>(T()) == T::value);

template <typename T>
concept bool_constant_like = requires { T::value; } &&
    same_as<bool, remove_const_t<decltype(T::value)>> &&
    convertible_to<T, bool> && equality_comparable_with<T, bool> &&
    (T() == T::value) && (static_cast<bool>(T()) == T::value);
} // namespace details::concepts

template <typename T>
concept integral_constant_like =
    details::concepts::integral_constant_like<remove_cv_t<T>>;

template <typename T>
concept bool_constant_like =
    details::concepts::bool_constant_like<remove_cv_t<T>>;

__DPL_DEFAULT_NAMESPACE_END
