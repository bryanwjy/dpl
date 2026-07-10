// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/concepts/derived_from.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
// Assumes IEEE floats, may need to change depending on platform
// TODO: Maybe move to math?
DPL_EXPORT template <typename>
struct floating_point_traits {};

DPL_EXPORT template <floating_point T>
struct floating_point_traits<T> {

    using type = T;

    static constexpr auto width = sizeof(T) * __DPL char_bit_v;

    static constexpr auto digits = static_cast<size_t>(
        __DPL countr_zero(__DPL bit_cast<bitset<width>>(static_cast<T>(1))) +
        1);

    static constexpr auto signbit = ~bitset<width>() << (width - 1);

    static constexpr auto mantissa_mask = bitset<width>(~bitset<digits - 1>());

    static constexpr auto exponent_mask = ~mantissa_mask ^ signbit;

    static constexpr auto has_hidden_bit = true;
};

DPL_EXPORT template <floating_point T>
requires (sizeof(T) == sizeof(bitset<80>) &&
    __DPL countr_zero(
        __DPL bit_cast<internal::xfp<sizeof(T)>>(static_cast<T>(1))
            .to_bitset()) == 63)
struct floating_point_traits<T> {
    using type = T;

    static constexpr auto width = 80zu;

    static constexpr auto digits = 64zu;

    static constexpr auto signbit = ~bitset<80>() << 79;

    static constexpr auto mantissa_mask = bitset<80>(~bitset<63>());

    static constexpr auto exponent_mask =
        ~mantissa_mask ^ bitset<80>(~bitset<64>());

    static constexpr auto has_hidden_bit = false;
};

DPL_EXPORT template <typename T>
concept floating_point_like =
    (floating_point<T> ||
        derived_from<T, internal::extended_floating_point<T>>) &&
    internal::has_floating_point_traits<T>;

} // namespace datapar

DPL_EXPORT using datapar::floating_point_like;
DPL_EXPORT using datapar::floating_point_traits;

DPL_DEFAULT_NAMESPACE_END
