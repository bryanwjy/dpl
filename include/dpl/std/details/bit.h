// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: always_keep

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT namespace details::bit {

#if DPL_HAS_BUILTIN(__builtin_popcountg)
// Nothing
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
    return __DPL details::bit::popcount(static_cast<unsigned int>(x));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned char x) noexcept {
    return __DPL details::bit::popcount(static_cast<unsigned int>(x));
}

#  if DPL_SUPPORTS_INT128
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(__uint128_t x) noexcept {
    static_assert(sizeof(unsigned long) * 2 == sizeof(__uint128_t));
    constexpr auto shift = (char_bit_v * sizeof(__uint128_t)) / 2;
    constexpr auto mask = (__uint128_t(1) << shift) - 1;
    return __DPL details::bit::popcount(static_cast<unsigned long>(x & mask)) +
        __DPL details::bit::popcount(static_cast<unsigned long>(x >> shift));
}
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
        return __DPL details::bit::cpopcount(x);
    }

    return __popcnt64(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned long x) noexcept {
    static_assert(sizeof(x) == 4, "MSVC");
    if consteval {
        return __DPL details::bit::cpopcount(x);
    }

    return __popcnt(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned int x) noexcept {
    if consteval {
        return __DPL details::bit::cpopcount(x);
    }

    return __popcnt(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned short x) noexcept {
    if consteval {
        return __DPL details::bit::cpopcount(x);
    }

    return __popcnt16(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr int popcount(unsigned char x) noexcept {
    return popcount(static_cast<unsigned int>(x));
}

#endif

#if DPL_HAS_BUILTIN(__builtin_popcountg)
inline constexpr size_t char_bit_v =
    __builtin_popcountg(static_cast<unsigned char>(-1));
#else
inline constexpr size_t char_bit_v =
    details::bit::pocount(static_cast<unsigned char>(-1));
#endif

template <typename T>
inline constexpr int unsigned_width_diff_v =
    char_bit_v * ((int)sizeof(unsigned) - (int)sizeof(T));

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
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline T rotl(T val, int count) noexcept {
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

template <unsigned_integral T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline T rotr(T val, int count) noexcept {
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
#endif

//

#if DPL_HAS_BUILTIN(__builtin_ctzg)
// Nothing
#elif DPL_HAS_BUILTIN(__builtin_ctz) & DPL_HAS_BUILTIN(__builtin_ctzl) & \
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
inline int rctz(unsigned long long x, unsigned long index = 0) noexcept {
    return _BitScanForward64(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline int rctz(unsigned long x, unsigned long index = 0) noexcept {
    return _BitScanForward(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline int rctz(unsigned int x, unsigned long index = 0) noexcept {
    return _BitScanForward(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline int rctz(unsigned short x, unsigned long index = 0) noexcept {
    return _BitScanForward(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline int rctz(unsigned char x, unsigned long index = 0) noexcept {
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
#endif

#if DPL_HAS_BUILTIN(__builtin_clzg)
// Nothing
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
inline int rclz(unsigned long long x, unsigned long index = 0) noexcept {
    return _BitScanReverse64(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline int rclz(unsigned long x, unsigned long index = 0) noexcept {
    return _BitScanReverse(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline int rclz(unsigned int x, unsigned long index = 0) noexcept {
    return _BitScanReverse(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline int rclz(unsigned short x, unsigned long index = 0) noexcept {
    return _BitScanReverse(&index, x) ? index : char_bit_v * sizeof(x);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline int rclz(unsigned char x, unsigned long index = 0) noexcept {
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

#endif

#if DPL_COMPILER_MSVC
extern "C" unsigned short _byteswap_ushort(unsigned short val);
extern "C" unsigned long _byteswap_ulong(unsigned long val);
extern "C" unsigned __int64 _byteswap_uint64(unsigned __int64 val);
#  pragma intrinsic(_byteswap_ushort)
#  pragma intrinsic(_byteswap_ulong)
#  pragma intrinsic(_byteswap_uint64)

template <integral T>
requires (sizeof(T) == sizeof(unsigned char))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
    return val;
}

template <integral T>
requires (sizeof(T) * char_bit_v == 16)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
    if constexpr (unsigned_integral<T>) {
        if consteval {
            return static_cast<unsigned short>((val << 8) | (val >> 8));
        } else {
            return _byteswap_ushort(val);
        }
    } else {
        return __DPL details::bit::byteswap(static_cast<unsigned short>(val));
    }
}

template <integral T>
requires (sizeof(T) * char_bit_v == 32)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto byteswap(T val) noexcept {
    if constexpr (unsigned_integral<T>) {
        if consteval {
            return (val << 24) | ((val << 8) & 0x00FF'0000) |
                ((val >> 8) & 0x0000'FF00) | (val >> 24);
        } else {
            return _byteswap_ulong(val);
        }
    } else {
        return __DPL details::bit::byteswap(static_cast<unsigned long>(val));
    }
}

template <integral T>
requires (sizeof(T) * char_bit_v == 64)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto byteswap(T val) noexcept {
    if constexpr (unsigned_integral<T>) {
        if consteval {
            return (val << 56) | ((val << 40) & 0x00FF'0000'0000'0000) |
                ((val << 24) & 0x0000'FF00'0000'0000) |
                ((val << 8) & 0x0000'00FF'0000'0000) |
                ((val >> 8) & 0x0000'0000'FF00'0000) |
                ((val >> 24) & 0x0000'0000'00FF'0000) |
                ((val >> 40) & 0x0000'0000'0000'FF00) | (val >> 56);
        } else {
            return _byteswap_uint64(val);
        }
    } else {
        return __DPL details::bit::byteswap(
            static_cast<unsigned __int64>(val));
    }
}

#else

template <integral T>
requires (sizeof(T) == sizeof(unsigned char))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
    return val;
}

#  if DPL_HAS_BUILTIN(__builtin_bswap16)
template <integral T>
requires (sizeof(T) * char_bit_v == 16)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
    return __builtin_bswap16(val);
}
#  endif

#  if DPL_HAS_BUILTIN(__builtin_bswap32)
template <integral T>
requires (sizeof(T) * char_bit_v == 32)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
    return __builtin_bswap32(val);
}
#  endif

#  if DPL_HAS_BUILTIN(__builtin_bswap64)
template <integral T>
requires (sizeof(T) * char_bit_v == 64)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
    return __builtin_bswap64(val);
}
#  endif

#  if DPL_SUPPORTS_INT128
template <integral T>
requires (sizeof(T) * char_bit_v == 128)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
#    if DPL_HAS_BUILTIN(__builtin_bswap128)
    return __builtin_bswap128(val);
#    else
    return (static_cast<T>(
                __DPL details::bit::byteswap(static_cast<uint64>(val)))
               << 64) |
        static_cast<T>(
            __DPL details::bit::byteswap(static_cast<uint64>(val >> 64)));
#    endif
}
#  endif

#endif
} // namespace details::bit

DPL_DEFAULT_NAMESPACE_END
