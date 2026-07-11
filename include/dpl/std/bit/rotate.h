// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/char_bit.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
#if DPL_COMPILER_MSVC
extern "C" unsigned char __cdecl _rotl8(unsigned char value, int shift);
extern "C" unsigned char __cdecl _rotr8(unsigned char value, int shift);
extern "C" unsigned short __cdecl _rotl16(unsigned short value, int shift);
extern "C" unsigned short __cdecl _rotr16(unsigned short value, int shift);
extern "C" unsigned int __cdecl _rotl(unsigned int value, int shift);
extern "C" unsigned int __cdecl _rotr(unsigned int value, int shift);
extern "C" unsigned __int64 __cdecl _rotl64(unsigned __int64 value, int shift);
extern "C" unsigned __int64 __cdecl _rotr64(unsigned __int64 value, int shift);
#  pragma intrinsic(_rotl8)
#  pragma intrinsic(_rotr8)
#  pragma intrinsic(_rotl16)
#  pragma intrinsic(_rotr16)
#  pragma intrinsic(_rotl)
#  pragma intrinsic(_rotr)
#  pragma intrinsic(_rotl64)
#  pragma intrinsic(_rotr64)
template <unsigned_integral T>
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
        if constexpr (sizeof(val) == sizeof(unsigned char)) {
            return _rotl8(val, count);
        } else if constexpr (sizeof(val) == sizeof(unsigned short)) {
            return _rotl16(val, count);
        } else if constexpr (sizeof(val) == sizeof(unsigned int)) {
            return _rotl(val, count);
        } else {
            static_assert(sizeof(val) == sizeof(unsigned __int64));
            return _rotl64(val, count);
        }
    }
}

template <unsigned_integral T>
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
        if constexpr (sizeof(val) == sizeof(unsigned char)) {
            return _rotr8(val, count);
        } else if constexpr (sizeof(val) == sizeof(unsigned short)) {
            return _rotr16(val, count);
        } else if constexpr (sizeof(val) == sizeof(unsigned int)) {
            return _rotr(val, count);
        } else {
            static_assert(sizeof(val) == sizeof(unsigned __int64));
            return _rotr64(val, count);
        }
    }
}

#else
template <unsigned_integral T>
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

template <unsigned_integral T>
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
__DPL_DEFAULT_NAMESPACE_END
