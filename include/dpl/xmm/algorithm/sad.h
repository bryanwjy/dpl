// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/operations/bitwise.h"
#    include "dpl/xmm/operations/reinterpret.h"

#    if DPL_SIMD_X86_SSE4_1
#      include "dpl/xmm/operations/compare/max.h"
#      include "dpl/xmm/operations/compare/min.h"
#    endif

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <same_as<uint8> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint8> lhs, vector<uint8> rhs) noexcept {
    return _mm_or_si128(
        _mm_subs_epu8(+lhs, +rhs), _mm_subs_epu8(+rhs, +lhs)); // |l - r|
}

template <same_as<uint16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint8> lhs, vector<uint8> rhs) noexcept {
    auto const ad = xmm::reinterpret<uint16>(xmm::sad<uint8>(lhs, rhs));
#  if DPL_SIMD_X86_SSE3
    return _mm_maddubs_epi16(+ad, +xmm::broadcast<uint8>(1));
#  else
    return xmm::add(xmm::bwand(ad, xmm::broadcast<uint16>(0xff)),
        xmm::bwshift_right<8>(ad));
#  endif
}

template <same_as<uint32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint8> lhs, vector<uint8> rhs) noexcept {
#  if DPL_SIMD_X86_AVX512VNNI & DPL_SIMD_X86_AVX512VL
    // TODO AVX also has vnni
    auto const ad = xmm::sad<uint8>(lhs, rhs);
    return _mm_dpbusd_epi32(+src, ad, +xmm::broadcast<uint8>(1));
#  else
    auto const u16 = xmm::sad<uint16>(lhs, rhs);
    return _mm_madd_epi16(+u16, +xmm::broadcast<uint16>(1));
#  endif
}

template <same_as<uint64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint8> lhs, vector<uint8> rhs) noexcept {
    return _mm_sad_epu8(+lhs, +rhs);
}

template <same_as<uint16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint16> lhs, vector<uint16> rhs) noexcept {
    return _mm_or_si128(
        _mm_subs_epu16(+lhs, +rhs), _mm_subs_epu16(+rhs, +lhs)); // |l - r|
}

template <same_as<uint32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint16> lhs, vector<uint16> rhs) noexcept {
    auto const ad = xmm::reinterpret<uint32>(xmm::sad<uint16>(lhs, rhs));
    return xmm::add(xmm::bwand(ad, xmm::broadcast<uint32>(0xffff)),
        xmm::bwshift_right<16>(ad));
}

template <same_as<uint64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint16> lhs, vector<uint16> rhs) noexcept {
    auto const u32 = xmm::reinterpret<uint64>(xmm::sad<uint32>(lhs, rhs));
    return xmm::add(xmm::bwand(u32, xmm::broadcast<uint64>(0xffffffff)),
        xmm::bwshift_right<32>(u32));
}

#  if DPL_SIMD_X86_SSE4_1
template <same_as<uint32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint32> lhs, vector<uint32> rhs) noexcept {
    return xmm::subtract(xmm::max(lhs, rhs), xmm::min(lhs, rhs));
}

template <same_as<uint64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL sad(vector<uint32> lhs, vector<uint32> rhs) noexcept {
    auto const ad = xmm::reinterpret<uint64>(xmm::sad<uint32>(lhs, rhs));
    return xmm::add(xmm::bwand(ad, xmm::broadcast<uint64>(0xffffffff)),
        xmm::bwshift_right<32>(ad));
}
#  endif

template <unsigned_integral To, signed_integral E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> sad(vector<E> lhs, vector<E> rhs) noexcept
requires requires(vector<make_unsigned_t<E>> arg) { xmm::sad<To>(arg, arg); }
{
    using uint_t = make_unsigned_t<E>;
    auto const bias = xmm::broadcast<uint_t>(dx::msb);
    return xmm::sad<To>(xmm::bwxor(bias, vector<uint_t>(+lhs)),
        xmm::bwxor(bias, vector<uint_t>(+rhs)));
}

template <unsigned_integral S, integral E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<S>
    DPL_VECTORCALL sad(vector<S> src, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::add(src, xmm::sad<S>(lhs, rhs)); }
{
    return xmm::add(src, xmm::sad<S>(lhs, rhs));
}

template <signed_integral S, integral E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<S>
    DPL_VECTORCALL sad(vector<S> src, vector<E> lhs, vector<E> rhs) noexcept
requires requires {
    xmm::sad(xmm::reinterpret<make_unsigned_t<S>>(src), lhs, rhs);
}
{
    return xmm::reinterpret<S>(
        xmm::sad(xmm::reinterpret<make_unsigned_t<S>>(src), lhs, rhs));
}

template <simd_element S, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<S>
    DPL_VECTORCALL sad(
        abi_tag, vector<S> src, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::sad(src, lhs, rhs); }
{
    return xmm::sad(src, lhs, rhs);
}
} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
