// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/floating_point_like.h"
#include "dpl/core/numbers/floating_point_traits.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
template <typename T>
concept binary_layout_floating_point = floating_point_like<T> && requires {
    floating_point_traits<T>::width;
    floating_point_traits<T>::digits;
    floating_point_traits<T>::signbit;
    floating_point_traits<T>::mantissa_mask;
    floating_point_traits<T>::leading_bit;
    floating_point_traits<T>::exponent_mask;
    floating_point_traits<T>::exponent_bias;
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
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::leading_bit>;
};
__DPL_DEFAULT_NAMESPACE_END
