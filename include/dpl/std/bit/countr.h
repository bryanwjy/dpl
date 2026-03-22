// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/char_bit.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::bit {
#if DPL_HAS_BUILTIN(__builtin_ctzg)

#  define __DPL_ctz(...)                                             \
      [](auto arg) {                                                 \
        return arg ? __builtin_ctzg(arg) : sizeof(arg) * char_bit_v; \
      }(__VA_ARGS__)

#elif DPL_HAS_BUILTIN(__builtin_ctz) & DPL_HAS_BUILTIN(__builtin_ctzl) &
DPL_HAS_BUILTIN(__builtin_ctzll)

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int ctz(unsigned long long x) noexcept {
    return x ? __builtin_ctzll(x) : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int ctz(unsigned long x) noexcept {
    return x ? __builtin_ctzl(x) : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int ctz(unsigned x) noexcept {
    return x ? __builtin_ctz(x) : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int ctz(unsigned short x) noexcept {
    return x ? __builtin_ctz(x) - unsigned_width_diff_v<unsigned short>
             : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int ctz(unsigned char x) noexcept {
    return x ? __builtin_ctz(x) - unsigned_width_diff_v<unsigned char>
             : char_bit_v * sizeof(x);
}

#  if DPL_SUPPORTS_INT128
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int ctz(__uint128_t x) noexcept {
    static_assert(sizeof(unsigned long) * 2 == sizeof(__uint128_t));
    constexpr auto shift = (char_bit_v * sizeof(__uint128_t)) / 2;

    constexpr auto mask = (__uint128_t(1) << shift) - 1;
    auto const lo = ctz(static_cast<unsigned long>(x >> shift));
    auto const hi = ctz(static_cast<unsigned long>(x & mask));
    auto const hi_bits = lo == shift ? hi : 0;
    return hi_bits + lo;
}

#    define __DPL_ctz(...) __DPL details::bit::ctz(__VA_ARGS__)
#  endif

#elif DPL_COMPILER_MSVC

extern "C" unsigned char _BitScanForward(
    unsigned long* Index, unsigned long Mask);
extern "C" unsigned char _BitScanForward64(
    unsigned long* Index, unsigned __int64 Mask);
#  pragma intrinsic(_BitScanForward64)
#  pragma intrinsic(_BitScanForward)

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int cctz(T x, unsigned int i = 1) noexcept {
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
int rctz(unsigned long long x, unsigned long index = 0) noexcept {
    return _BitScanForward64(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rctz(unsigned long x, unsigned long index = 0) noexcept {
    return _BitScanForward(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rctz(unsigned int x, unsigned long index = 0) noexcept {
    return _BitScanForward(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rctz(unsigned short x, unsigned long index = 0) noexcept {
    return _BitScanForward(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
int rctz(unsigned char x, unsigned long index = 0) noexcept {
    return _BitScanForward(&index, x) ? index : char_bit_v * sizeof(x);
}

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int ctz(T x) noexcept {
    if consteval {
        return cctz(x);
    } else {
        return rctz(x);
    }
}

#  define __DPL_ctz(...) __DPL details::bit::ctz(__VA_ARGS__)
#endif
} // namespace details::bit

DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countr_zero(T val) noexcept {
    return __DPL_ctz(val);
}

DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countr_one(T val) noexcept {
    return __DPL_ctz(static_cast<T>(~val));
}

#undef __DPL_ctz
DPL_DEFAULT_NAMESPACE_END
