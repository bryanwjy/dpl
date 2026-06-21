// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/structured_bindings.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/equality_comparable.h"
#  include "dpl/std/type_traits/remove_cv.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept bitset_type = is_bitset_v<T>;

DPL_EXPORT template <typename T>
concept integral_bitset_type =
    bitset_type<T> && integral<typename remove_cv_t<T>::underlying_type>;

namespace details {
template <typename T>
concept bitset_constant_like = requires { T::value; } &&
    bitset_type<decltype(T::value)> && convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> &&
    bool_constant<T() == T::value>::value &&
    bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value;
} // namespace details

DPL_EXPORT template <typename T>
concept bitset_constant_like = details::bitset_constant_like<remove_cv_t<T>>;

template <bitset_type auto T>
using bitset_constant = integral_constant<decltype(T), T>;

DPL_DEFAULT_NAMESPACE_END
