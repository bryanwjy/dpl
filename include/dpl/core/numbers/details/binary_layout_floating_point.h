// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/details/fwd.h"

#include "dpl/core/numbers/details/floating_point_like.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT namespace details::numbers {

template <typename T>
concept binary_layout_floating_point = floating_point_like<T> && requires {
    floating_point_traits<T>::width;
    floating_point_traits<T>::digits;
    floating_point_traits<T>::signbit;
    floating_point_traits<T>::mantissa_mask;
    floating_point_traits<T>::exponent_mask;
    floating_point_traits<T>::exponent_bias;
    floating_point_traits<T>::has_hidden_bit;
    floating_point_traits<T>::has_signaling_nan;
    floating_point_traits<T>::has_denormal;
    floating_point_traits<T>::has_infinity;
    floating_point_traits<T>::radix;
    requires (floating_point_traits<T>::radix == 2);
    typename size_constant<floating_point_traits<T>::width>;
    typename size_constant<floating_point_traits<T>::exponent_bias>;
    typename size_constant<floating_point_traits<T>::digits>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::signbit>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::mantissa_mask>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::exponent_mask>;
    typename bool_constant<floating_point_traits<T>::has_hidden_bit>;
    typename bool_constant<floating_point_traits<T>::has_signaling_nan>;
    typename bool_constant<floating_point_traits<T>::has_quiet_nan>;
    typename bool_constant<floating_point_traits<T>::has_denormal>;
    typename bool_constant<floating_point_traits<T>::has_infinity>;
};

} // namespace details::numbers
DPL_DEFAULT_NAMESPACE_END
