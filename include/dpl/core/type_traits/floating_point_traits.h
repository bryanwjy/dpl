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

namespace internal {
template <size_t N>
struct xfp {
    static_assert(80 % __DPL char_bit_v == 0);
    static constexpr auto bytes = 80zu / __DPL char_bit_v;
    struct val_t {
        unsigned char data[bytes];
    } val;
    struct padding_t {
        unsigned char data[N - bytes];
    } padding;
    consteval bitset<80> to_bitset() const {
        xfp copy{};
        copy.val = this->val;
        return __DPL bit_cast<bitset<80>>(copy);
    }
};
} // namespace internal

template <typename T>
concept extended_precision_type = floating_point<T> &&
    (sizeof(T) == sizeof(bitset<80>) &&
        __DPL countr_zero(
            __DPL bit_cast<internal::xfp<sizeof(T)>>(static_cast<T>(1))
                .to_bitset()) == 63);

DPL_EXPORT template <extended_precision_type T>
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

template <typename T>
concept has_floating_point_traits = requires {
    typename floating_point_traits<T>::type;
    floating_point_traits<T>::width;
    floating_point_traits<T>::digits;
    floating_point_traits<T>::signbit;
    floating_point_traits<T>::mantissa_mask;
    floating_point_traits<T>::exponent_mask;
    floating_point_traits<T>::has_hidden_bit;
    typename size_constant<floating_point_traits<T>::width>;
    typename size_constant<floating_point_traits<T>::digits>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::signbit>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::mantissa_mask>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::exponent_mask>;
    typename integral_constant<bool, floating_point_traits<T>::has_hidden_bit>;
};

// Not exported for now
template <typename T>
class extended_floating_point {};

DPL_EXPORT template <typename T>
concept floating_point_like =
    (floating_point<T> || derived_from<T, extended_floating_point<T>>) &&
    has_floating_point_traits<T>;

} // namespace datapar

DPL_EXPORT using datapar::floating_point_like;
DPL_EXPORT using datapar::floating_point_traits;

DPL_DEFAULT_NAMESPACE_END
