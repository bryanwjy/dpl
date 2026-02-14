// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_element.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/equality_comparable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

namespace internal {

template <typename T>
concept immediate_like = convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> &&
    bool_constant<T() == T::value>::value &&
    bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value;

template <typename T, typename E>
concept immediate_like_of =
    simd_element<E> && convertible_to<T, E> && requires {
        typename integral_constant<E, static_cast<E>(T())>;
        requires immediate_like<integral_constant<E, static_cast<E>(T())>>;
    };

} // namespace internal
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
