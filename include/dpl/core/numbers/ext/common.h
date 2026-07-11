// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/floating_point_traits.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/stddef/types.h"
#  include "dpl/std/type_traits/common_type.h"
#  include "dpl/std/utility/bitset.h" // IWYU pragma: keep
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace ext {

#if !DPL_SUPPORTS_BFLOAT16
class bfloat16_t;
using bfloat16 = bfloat16_t;
#else
using bfloat16 = __DPL bfloat16;
#endif
#if !DPL_SUPPORTS_FLOAT16
class float16_t;
using float16 = float16_t;
#else
using float16 = __DPL float16;
#endif

inline namespace literals {}
} // namespace ext

namespace ext_literals = ext::literals; // NOLINT(misc-unused-alias-decls)

#if !DPL_SUPPORTS_BFLOAT16
template <floating_point T>
struct common_type<ext::bfloat16, T> {
    using type = T;
};
template <floating_point T>
struct common_type<T, ext::bfloat16> {
    using type = T;
};
template <integral T>
struct common_type<T, ext::bfloat16> {
    using type = ext::bfloat16;
};
template <integral T>
struct common_type<ext::bfloat16, T> {
    using type = ext::bfloat16;
};

template <>
struct floating_point_traits<ext::bfloat16> {
    using type = ext::bfloat16;

    static constexpr auto width = 16zu;
    static constexpr auto digits = 8zu;
    static constexpr auto signbit = bitset<16>(1u << 15);
    static constexpr auto mantissa_mask =
        __DPL truncate<16>(floating_point_traits<float>::mantissa_mask >> 16);
    static constexpr auto exponent_mask =
        __DPL truncate<16>(floating_point_traits<float>::exponent_mask >> 16);
    static constexpr auto exponent_bias =
        ((1 << __DPL popcount(exponent_mask)) - 1);
    static constexpr auto has_hidden_bit = true;
    static constexpr auto has_denormal = true;
    static constexpr auto has_infinity = true;
    static constexpr auto has_quiet_nan = true;
    static constexpr auto has_signaling_nan = true;
};
#endif

#if !DPL_SUPPORTS_FLOAT16
template <floating_point T>
struct common_type<ext::float16, T> {
    using type = T;
};
template <floating_point T>
struct common_type<T, ext::float16> {
    using type = T;
};
template <integral T>
struct common_type<T, ext::float16> {
    using type = ext::float16;
};
template <integral T>
struct common_type<ext::float16, T> {
    using type = ext::float16;
};

template <>
struct floating_point_traits<ext::float16> {
    using type = ext::float16;

    static constexpr auto width = 16zu;
    static constexpr auto digits = 11zu;
    static constexpr auto signbit = bitset<16>(1u << 15);
    static constexpr auto mantissa_mask = bitset<16>((1u << (digits - 1)) - 1);
    static constexpr auto exponent_mask = ~mantissa_mask ^ signbit;
    static constexpr auto exponent_bias =
        ((1 << __DPL popcount(exponent_mask)) - 1);
    static constexpr auto has_hidden_bit = true;
    static constexpr auto has_denormal = true;
    static constexpr auto has_infinity = true;
    static constexpr auto has_quiet_nan = true;
    static constexpr auto has_signaling_nan = true;
};
#endif
__DPL_DEFAULT_NAMESPACE_END
