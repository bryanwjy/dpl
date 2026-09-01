// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename>
struct floating_point_traits {};

// Assumes IEEE floats, may need to change depending on platform
template <floating_point T>
struct floating_point_traits<T> {

    using type = T;

    static constexpr auto width = __DPL type_bit_v<T>;

    static constexpr auto digits = static_cast<size_t>(
        __DPL countr_zero(__DPL to_bit_representation(static_cast<T>(1))) +
        1);

    static constexpr auto signbit =
        bit_representation_t<T>(__DPL high_bits, 1);

    static constexpr auto mantissa_mask =
        bit_representation_t<T>(__DPL low_bits, digits - 1);

    static constexpr auto leading_bit = bit_representation_t<T>();

    static constexpr auto exponent_mask = ~mantissa_mask ^ signbit;

    static constexpr auto exponent_bias =
        static_cast<int>(__DPL to_underlying(exponent_mask >> digits));

    static constexpr auto radix = 2zu;
};

template <floating_point T>
requires same_as<bit_representation_t<T>, bitset<80>>
struct floating_point_traits<T> {
    using type = T;

    static constexpr auto width = 80zu;

    static constexpr auto digits = 64zu;

    static constexpr auto signbit = bitset<80>(__DPL high_bits, 1);

    static constexpr auto mantissa_mask = bitset<80>(__DPL low_bits, 63);

    static constexpr auto leading_bit = bitset<80>(__DPL bit_range, 63, 1);

    static constexpr auto exponent_mask =
        ~(signbit | mantissa_mask | leading_bit);

    static constexpr auto exponent_bias =
        static_cast<int>(__DPL to_underlying(exponent_mask >> digits));

    static constexpr auto radix = 2zu;
};

__DPL_DEFAULT_NAMESPACE_END
