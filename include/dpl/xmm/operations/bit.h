// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/arithmetic.h"
#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/std/bit/countl.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/broadcast.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R>
    DPL_VECTORCALL bit_drop(simd<L> lhs, simd<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result = _mm_andnot_si128(
        +xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(simd<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<R>
    DPL_VECTORCALL bit_drop(mask<L> lhs, mask<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result = _mm_andnot_si128(
        +xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(mask<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R>
    DPL_VECTORCALL bit_drop(mask<L> lhs, simd<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result = _mm_andnot_si128(
        +xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(simd<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R>
    DPL_VECTORCALL bit_keep(simd<L> lhs, simd<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result =
        _mm_and_si128(+xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(simd<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<R>
    DPL_VECTORCALL bit_keep(mask<L> lhs, mask<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result =
        _mm_and_si128(+xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(mask<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R>
    DPL_VECTORCALL bit_keep(mask<L> lhs, simd<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result =
        _mm_and_si128(+xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(simd<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R>
    DPL_VECTORCALL bit_fill(simd<L> lhs, simd<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result =
        _mm_or_si128(+xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(simd<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<R>
    DPL_VECTORCALL bit_fill(mask<L> lhs, mask<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result =
        _mm_or_si128(+xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(mask<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R>
    DPL_VECTORCALL bit_fill(mask<L> lhs, simd<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const result =
        _mm_or_si128(+xmm::reinterpret<bit>(lhs), +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(simd<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R>
    DPL_VECTORCALL bit_stencil(simd<L> lhs, simd<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const all = _mm_set1_epi32(-1);
    auto const result =
        _mm_or_si128(_mm_xor_si128(all, +xmm::reinterpret<bit>(lhs)),
            +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(simd<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<R>
    DPL_VECTORCALL bit_stencil(mask<L> lhs, mask<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const all = _mm_set1_epi32(-1);
    auto const result =
        _mm_or_si128(_mm_xor_si128(all, +xmm::reinterpret<bit>(lhs)),
            +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(mask<bit>(result));
}

template <simd_element L, common_size_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R>
    DPL_VECTORCALL bit_stencil(mask<L> lhs, simd<R> rhs) noexcept {
    using bit = unsigned_representation_t<L>;
    auto const all = [](__m128i val) { return _mm_cmpeq_epi32(val, val); }(
                         _mm_undefined_si128());
    auto const result =
        _mm_or_si128(_mm_xor_si128(all, +xmm::reinterpret<bit>(lhs)),
            +xmm::reinterpret<bit>(rhs));
    return xmm::reinterpret<R>(simd<bit>(result));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL bit_drop(abi_tag, simd<L> lhs, simd<R> rhs) noexcept
requires requires { xmm::bit_drop(lhs, rhs); }
{
    return xmm::bit_drop(lhs, rhs);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL bit_keep(abi_tag, simd<L> lhs, simd<R> rhs) noexcept
requires requires { xmm::bit_keep(lhs, rhs); }
{
    return xmm::bit_keep(lhs, rhs);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL bit_fill(abi_tag, simd<L> lhs, simd<R> rhs) noexcept
requires requires { xmm::bit_fill(lhs, rhs); }
{
    return xmm::bit_fill(lhs, rhs);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL bit_stencil(
    abi_tag, simd<L> lhs, simd<R> rhs) noexcept
requires requires { xmm::bit_stencil(lhs, rhs); }
{
    return xmm::bit_stencil(lhs, rhs);
}

namespace details {
[[maybe_unused]] alignas(16) inline constexpr char popcnt_lut[]{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<signed_representation_t<E>>
    DPL_VECTORCALL popcount(simd<E> val) noexcept {
    if constexpr (sizeof(E) == 1) {
#if DPL_SIMD_X86_AVX512BITALG && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi8(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#else
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::popcnt_lut));
        auto const mask = _mm_set1_epi8(0x0f);
        auto const lo = _mm_and_si128(+val, mask);
        auto const hi = _mm_and_si128(_mm_srli_epi16(+val, 4), mask);
        return _mm_add_epi8(
            _mm_shuffle_epi8(lut, lo), _mm_shuffle_epi8(lut, hi));
#endif
    } else if constexpr (sizeof(E) == 2) {
#if DPL_SIMD_X86_AVX512BITALG && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi16(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#else
        using subbit = bit_type_t<char_bit_v * sizeof(E) / 2>;
        return _mm_maddubs_epi16(+xmm::popcount(xmm::reinterpret<subbit>(val)),
            _mm_set1_epi16(0x0101));
#endif
    } else if constexpr (sizeof(E) == 4) {
#if DPL_SIMD_X86_AVX512VPOPCNTDQ && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi32(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#else
        using subbit = bit_type_t<char_bit_v * sizeof(E) / 2>;
        return _mm_maddubs_epi16(+xmm::popcount(xmm::reinterpret<subbit>(val)),
            _mm_set1_epi32(0x00010001));
#endif
    } else {
        static_assert(sizeof(E) == 8);
#if DPL_SIMD_X86_AVX512VPOPCNTDQ && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi64(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#else
        using subbit = bit_type_t<char_bit_v * sizeof(E) / 8>;
        return _mm_sad_epu8(
            +xmm::popcount(xmm::reinterpret<subbit>(val)), _mm_setzero_si128());
#endif
    }
}

template <simd_element E>
requires (sizeof(E) == 8)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<signed_representation_t<E>>
    DPL_VECTORCALL countl_zero(simd<E> val) noexcept {
#if DPL_SIMD_X86_AVX512CD && DPL_SIMD_X86_AVX512VL
    return _mm_lzcnt_epi64(+xmm::reinterpret<signed_representation_t<E>>(val));
#else
    using sint = signed_representation_t<E>;
    auto const vval = xmm::reinterpret<sint>(val);
    return xmm::initialize<sint>( __DPL countl_zero(xmm::extract<0>(vval)),
        __DPL countl_zero(xmm::extract<1>(vval)));
#endif
}

template <simd_element E>
requires (sizeof(E) == 4)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<signed_representation_t<E>>
    DPL_VECTORCALL countl_zero(simd<E> val) noexcept {
#if DPL_SIMD_X86_AVX512CD && DPL_SIMD_X86_AVX512VL
    return _mm_lzcnt_epi32(+xmm::reinterpret<signed_representation_t<E>>(val));
#else
    using sint = signed_representation_t<E>;
    auto const vval = xmm::reinterpret<sint>(val);
    return xmm::initialize<sint>( __DPL countl_zero(xmm::extract<0>(vval)),
        __DPL countl_zero(xmm::extract<1>(vval)),
        __DPL countl_zero(xmm::extract<3>(vval)),
        __DPL countl_zero(xmm::extract<4>(vval)));
#endif
}

template <simd_element E>
requires (sizeof(E) == 2)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<signed_representation_t<E>>
    DPL_VECTORCALL countl_zero(simd<E> val) noexcept {
    auto vval = +xmm::reinterpret<signed_representation_t<E>>(val);
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 1));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 2));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 4));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 8));
    return _mm_sub_epi16(_mm_set1_epi16(16), xmm::popcount(vval));
}

template <simd_element E>
requires (sizeof(E) == 1)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<signed_representation_t<E>>
    DPL_VECTORCALL countl_zero(simd<E> val) noexcept {
    auto vval = +xmm::reinterpret<signed_representation_t<E>>(val);
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 1));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 2));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 4));
    return _mm_sub_epi8(_mm_set1_epi8(8), xmm::popcount(vval));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<signed_representation_t<E>>
    DPL_VECTORCALL countl_one(simd<E> val) noexcept {
    using sbit = signed_representation_t<E>;
    auto vval = +xmm::reinterpret<sbit>(val);
    return xmm::countl_zero(
        simd<sbit>(_mm_xor_si128(vval, _mm_set1_epi32(-1))));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<signed_representation_t<E>>
    DPL_VECTORCALL countr_zero(simd<E> val) noexcept {
    if constexpr (!unsigned_integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::countr_zero(xmm::reinterpret<bit>(val));
    } else {
        auto const nval = xmm::subtract(simd<E>(_mm_setzero_si128()), val);
        return xmm::popcount(xmm::subtract(
            simd<E>(_mm_and_si128(+val, +nval)), xmm::broadcast<E>(1u)));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<signed_representation_t<E>>
    DPL_VECTORCALL countr_one(simd<E> val) noexcept {
    using bit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<bit>(val);
    return xmm::countr_zero(
        simd<bit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

namespace details {

alignas(16) inline constexpr char byteswap_lut_epi64[]{
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8};

alignas(16) inline constexpr char byteswap_lut_epi32[]{
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};

alignas(16) inline constexpr char byteswap_lut_epi16[]{
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};

} // namespace details

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
inline simd<E>
    DPL_VECTORCALL byteswap(simd<E> val) noexcept {
    if constexpr (!same_as<native_vector_t<E>, __m128i>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(xmm::byteswap(xmm::reinterpret<bit>(val)));
    } else if constexpr (sizeof(E) == 8) {
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::byteswap_lut_epi64));
        return _mm_shuffle_epi8(+val, lut);
    } else if constexpr (sizeof(E) == 4) {
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::byteswap_lut_epi32));
        return _mm_shuffle_epi8(+val, lut);
    } else if constexpr (sizeof(E) == 2) {
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::byteswap_lut_epi16));
        return _mm_shuffle_epi8(+val, lut);
    } else {
        static_assert(sizeof(E) == 1);
        return val;
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL popcount(abi_tag, simd<E> val) noexcept
requires requires { xmm::popcount(val); }
{
    return xmm::popcount(val);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_zero(abi_tag, simd<E> val) noexcept
requires requires { xmm::countl_zero(val); }
{
    return xmm::countl_zero(val);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_zero(abi_tag, simd<E> val) noexcept
requires requires { xmm::countl_zero(val); }
{
    return xmm::countl_zero(val);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL byteswap(abi_tag, simd<E> val) noexcept
requires requires { xmm::byteswap(val); }
{
    return xmm::byteswap(val);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
