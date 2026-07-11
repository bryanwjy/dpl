// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::bit {
#if DPL_HAS_BUILTIN(__builtin_popcountg)

#  define __DPL_popcount(...) __builtin_popcountg(__VA_ARGS__)

#elif DPL_HAS_BUILTIN(__builtin_popcount) & \
    DPL_HAS_BUILTIN(__builtin_popcountll)

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned long long x) noexcept {
    return __builtin_popcountll(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned long x) noexcept {
    return __builtin_popcountl(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned int x) noexcept {
    return __builtin_popcount(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned short x) noexcept {
    return __VDPL details::bit::popcount(static_cast<unsigned int>(x));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned char x) noexcept {
    return __VDPL details::bit::popcount(static_cast<unsigned int>(x));
}

#  if DPL_SUPPORTS_INT128
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(__uint128_t x) noexcept {
    static_assert(sizeof(unsigned long) * 2 == sizeof(__uint128_t));
    constexpr auto shift = (char_bit_v * sizeof(__uint128_t)) / 2;
    constexpr auto mask = (__uint128_t(1) << shift) - 1;
    return __VDPL details::bit::popcount(static_cast<unsigned long>(x & mask)) +
        __VDPL details::bit::popcount(static_cast<unsigned long>(x >> shift));
}

#    define __DPL_popcount(...) __VDPL details::bit::popcount(__VA_ARGS__)
#  endif

#elif DPL_COMPILER_MSVC

extern "C" unsigned short __popcnt16(unsigned short);
extern "C" unsigned int __popcnt(unsigned int);
extern "C" unsigned __int64 __popcnt64(unsigned __int64);
#  pragma intrinsic(__popcnt16)
#  pragma intrinsic(__popcnt)
#  pragma intrinsic(__popcnt64)

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int cpopcount(T x, int c = 0) noexcept {
    return !x ? c : cpopcount(x & (x - 1), c + 1);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned long long x) noexcept {
    if consteval {
        return __VDPL details::bit::cpopcount(x);
    }

    return __popcnt64(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned long x) noexcept {
    static_assert(sizeof(x) == 4, "MSVC");
    if consteval {
        return __VDPL details::bit::cpopcount(x);
    }

    return __popcnt(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned int x) noexcept {
    if consteval {
        return __VDPL details::bit::cpopcount(x);
    }

    return __popcnt(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned short x) noexcept {
    if consteval {
        return __VDPL details::bit::cpopcount(x);
    }

    return __popcnt16(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned char x) noexcept {
    return popcount(static_cast<unsigned int>(x));
}

#  define __DPL_popcount(...) __VDPL details::bit::popcount(__VA_ARGS__)
#endif
} // namespace details::bit

template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int popcount(T val) noexcept {
    return __DPL_popcount(val);
}

#undef __DPL_popcount
__DPL_DEFAULT_NAMESPACE_END
