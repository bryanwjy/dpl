// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/char_bit.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/details/bit.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

#if DPL_COMPILER_MSVC
DPL_EXPORT template <unsigned_integral T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr T rotl(T val, int count) noexcept {
    if consteval {
        constexpr auto digits = sizeof(T) * __DPL char_bit_v;
        count %= digits;
        if (count == 0)
            return val;

        if (count > 0) {
            return (val << count) | (val >> (digits - count));
        }

        return (val >> -count) | (val << (digits + count));
    } else {
        return __DPL details::bit::rotl(val, count);
    }
}

DPL_EXPORT template <unsigned_integral T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr T rotr(T val, int count) noexcept {
    if consteval {
        constexpr auto digits = sizeof(T) * __DPL char_bit_v;
        count %= digits;
        if (count == 0)
            return val;

        if (count > 0) {
            return (val >> count) | (val << (digits - count));
        }

        return (val << -count) | (val >> (digits + count));
    } else {
        return __DPL details::bit::rotr(val, count);
    }
}

#else
DPL_EXPORT template <unsigned_integral T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr T rotl(T val, int count) noexcept {
    constexpr auto digits = sizeof(T) * __DPL char_bit_v;
    count %= digits;
    if (count == 0)
        return val;

    if (count > 0) {
        return (val << count) | (val >> (digits - count));
    }

    return (val >> -count) | (val << (digits + count));
}

DPL_EXPORT template <unsigned_integral T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr T rotr(T val, int count) noexcept {
    constexpr auto digits = sizeof(T) * __DPL char_bit_v;
    count %= digits;
    if (count == 0)
        return val;

    if (count > 0) {
        return (val >> count) | (val << (digits - count));
    }

    return (val << -count) | (val >> (digits + count));
}
#endif
DPL_DEFAULT_NAMESPACE_END
