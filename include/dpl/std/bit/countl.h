// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#endif

#include "dpl/std/bit/char_bit.h"
#include "dpl/std/bit/popcount.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::bit {
#if DPL_HAS_BUILTIN(__builtin_clzg)

#  define __DPL_clz(...) __builtin_clzg(__VA_ARGS__)

#elif DPL_HAS_BUILTIN(__builtin_clz) & DPL_HAS_BUILTIN(__builtin_clzl) &
DPL_HAS_BUILTIN(__builtin_clzll)

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int clz(unsigned long long x) noexcept {
    return x ? __builtin_clzll(x) : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int clz(unsigned long x) noexcept {
    return x ? __builtin_clzl(x) : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int clz(unsigned x) noexcept {
    return x ? __builtin_clz(x) : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int clz(unsigned short x) noexcept {
    return x ? __builtin_clz(x) - unsigned_width_diff_v<unsigned short>
             : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int clz(unsigned char x) noexcept {
    return x ? __builtin_clz(x) - unsigned_width_diff_v<unsigned char>
             : char_bit_v * sizeof(x);
}

#  if DPL_SUPPORTS_INT128
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int clz(__uint128_t x) noexcept {
    static_assert(sizeof(unsigned long) * 2 == sizeof(__uint128_t));
    constexpr auto shift = (char_bit_v * sizeof(__uint128_t)) / 2;
    constexpr auto mask = (__uint128_t(1) << shift) - 1;
    auto const hi = clz(static_cast<unsigned long>(x & mask));
    auto const lo = clz(static_cast<unsigned long>(x >> shift));
    auto const lo_bits = hi == shift ? lo : 0;
    return hi + lo_bits;
}

#    define __DPL_clz(...) __DPL details::bit::clz(__VA_ARGS__)
#  endif

#elif DPL_COMPILER_MSVC

extern "C" unsigned char _BitScanReverse(
    unsigned long* Index, unsigned long Mask);
extern "C" unsigned char _BitScanReverse64(
    unsigned long* Index, unsigned __int64 Mask);
#  pragma intrinsic(_BitScanReverse64)
#  pragma intrinsic(_BitScanReverse)

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int cclz(T x, unsigned int i = 1) noexcept {
    return x ? (x &= -(int64_t)x),
           ((!(x & 0xFFFFFFFFull) << 5 | !(x & 0xFFFF0000FFFFull) << 4 |
                !(x & 0xFF00FF00FF00FFull) << 3 |
                !(x & 0xF0F0F0F0F0F0F0Full) << 2 |
                !(x & 0x3333333333333333ull) << 1 |
                !(x & 0x5555555555555555ull)) +
               !x)
             : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rclz(unsigned long long x, unsigned long index = 0) noexcept {
    return _BitScanReverse64(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rclz(unsigned long x, unsigned long index = 0) noexcept {
    return _BitScanReverse(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rclz(unsigned int x, unsigned long index = 0) noexcept {
    return _BitScanReverse(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rclz(unsigned short x, unsigned long index = 0) noexcept {
    return _BitScanReverse(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rclz(unsigned char x, unsigned long index = 0) noexcept {
    return _BitScanReverse(&index, x) ? index : char_bit_v * sizeof(x);
}

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int clz(T x) noexcept {
    if consteval {
        return cclz(x);
    } else {
        return rclz(x);
    }
}

#  define __DPL_clz(...) __DPL details::bit::clz(__VA_ARGS__)
#endif
} // namespace details::bit

template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countl_zero(T val) noexcept {
    return __DPL_clz(val);
}

template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countl_one(T val) noexcept {
    return __DPL_clz(static_cast<T>(~val));
}

#undef __DPL_clz
DPL_DEFAULT_NAMESPACE_END
