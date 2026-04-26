// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_element.h"

#if !DPL_MODULES
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/type_traits/constants.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
inline constexpr bool enable_immediate_mask = false;

template <typename T>
concept immediate_mask_like = enable_immediate_mask<T> &&
    integral_constant_like<T> && unsigned_integral<typename T::value_type> &&
    requires { typename integral_constant<size_t, T::width>; } &&
    (T::width < sizeof(typename T::value_type) * char_bit_v);
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
