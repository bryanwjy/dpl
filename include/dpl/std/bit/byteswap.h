// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/bit/bit_type.h"
#include "dpl/std/bit/char_bit.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::bit {
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
    return (static_cast<T>(__DPL details::bit::byteswap(
                static_cast<bit_type_t<64>>(val)))
               << 64) |
        static_cast<T>(__DPL details::bit::byteswap(
            static_cast<bit_type_t<64>>(val >> 64)));
#    endif
}
#  endif

#endif

} // namespace details::bit

template <integral T>
requires requires(T val) { __DPL details::bit::byteswap(val); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T byteswap(T val) noexcept {
    return __DPL details::bit::byteswap(val);
}

DPL_DEFAULT_NAMESPACE_END
