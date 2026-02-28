// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/arithmetic.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/type_traits/common_bits_type.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/broadcast.h"
#  include "dpl/xmm/basic/reinterpret.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> bit_drop(
    abi_tag tag, simd<L> lhs, simd<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_andnot_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<common_bits_type_t<L, R>>(tag, simd<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> bit_drop(
    abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_andnot_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<common_bits_type_t<L, R>>(tag, mask<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R> bit_drop(abi_tag tag, mask<L> lhs, simd<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_andnot_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<R>(tag, simd<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> bit_keep(
    abi_tag tag, simd<L> lhs, simd<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_and_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<common_bits_type_t<L, R>>(tag, simd<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> bit_keep(
    abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_and_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<common_bits_type_t<L, R>>(tag, mask<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R> bit_keep(abi_tag tag, mask<L> lhs, simd<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_and_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<R>(tag, simd<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> bit_fill(
    abi_tag tag, simd<L> lhs, simd<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_or_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<common_bits_type_t<L, R>>(tag, simd<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> bit_fill(
    abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_or_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<common_bits_type_t<L, R>>(tag, mask<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R> bit_fill(abi_tag tag, mask<L> lhs, simd<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const result = _mm_or_si128(
        +xmm::reinterpret<bit>(tag, lhs), +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<R>(tag, simd<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> bit_stencil(
    abi_tag tag, simd<L> lhs, simd<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const all = [](__m128i val) { return _mm_cmpeq_epi32(val, val); }(
                         _mm_undefined_si128());
    auto const result =
        _mm_or_si128(_mm_xor_si128(all, +xmm::reinterpret<bit>(tag, lhs)),
            +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<common_bits_type_t<L, R>>(tag, simd<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> bit_stencil(
    abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const all = [](__m128i val) { return _mm_cmpeq_epi32(val, val); }(
                         _mm_undefined_si128());
    auto const result =
        _mm_or_si128(_mm_xor_si128(all, +xmm::reinterpret<bit>(tag, lhs)),
            +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<common_bits_type_t<L, R>>(tag, mask<bit>(result));
}

template <simd_element L, common_bits_with<L> R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<R> bit_stencil(abi_tag tag, mask<L> lhs, simd<R> rhs) noexcept {
    using bit = to_unsigned_integral_t<L>;
    auto const all = [](__m128i val) { return _mm_cmpeq_epi32(val, val); }(
                         _mm_undefined_si128());
    auto const result =
        _mm_or_si128(_mm_xor_si128(all, +xmm::reinterpret<bit>(tag, lhs)),
            +xmm::reinterpret<bit>(tag, rhs));
    return xmm::reinterpret<R>(tag, simd<bit>(result));
}

namespace details {
[[maybe_unused]] alignas(16) inline constexpr char popcnt_lut[]{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
inline simd<to_signed_integral_t<E>> popcount(
    abi_tag tag, simd<E> val) noexcept {
    if constexpr (sizeof(E) == 1) {
#if DPL_SIMD_X86_AVX512BITALG && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi8(
            +xmm::reinterpret<to_signed_integral_t<E>>(tag, val));
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
            +xmm::reinterpret<to_signed_integral_t<E>>(tag, val));
#else
        using subbit = bit_type_t<char_bit_v * sizeof(E) / 2>;
        return _mm_maddubs_epi16(
            +xmm::popcount(tag, xmm::reinterpret<subbit>(tag, val)),
            _mm_set1_epi16(0x0101));
#endif
    } else if constexpr (sizeof(E) == 4) {
#if DPL_SIMD_X86_AVX512VPOPCNTDQ && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi32(
            +xmm::reinterpret<to_signed_integral_t<E>>(tag, val));
#else
        using subbit = bit_type_t<char_bit_v * sizeof(E) / 2>;
        return _mm_maddubs_epi16(
            +xmm::popcount(tag, xmm::reinterpret<subbit>(tag, val)),
            _mm_set1_epi32(0x00010001));
#endif
    } else {
        static_assert(sizeof(E) == 8);
#if DPL_SIMD_X86_AVX512VPOPCNTDQ && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi64(
            +xmm::reinterpret<to_signed_integral_t<E>>(tag, val));
#else
        using subbit = bit_type_t<char_bit_v * sizeof(E) / 8>;
        return _mm_sad_epu8(
            +xmm::popcount(tag, xmm::reinterpret<subbit>(tag, val)),
            _mm_setzero_si128());
#endif
    }
}

#if DPL_SIMD_X86_AVX512CD && DPL_SIMD_X86_AVX512VL
template <simd_element E>
requires (sizeof(E) == 4)
inline simd<to_signed_integral_t<E>> countl_zero(
    abi_tag tag, simd<E> val) noexcept {
    return _mm_lzcnt_epi32(
        +xmm::reinterpret<to_signed_integral_t<E>>(tag, val));
}

template <simd_element E>
requires (sizeof(E) == 8)
inline simd<to_signed_integral_t<E>> countl_zero(
    abi_tag tag, simd<E> val) noexcept {
    return _mm_lzcnt_epi64(
        +xmm::reinterpret<to_signed_integral_t<E>>(tag, val));
}

template <simd_element E>
requires (sizeof(E) == 8 || sizeof(E) == 4)
inline simd<to_signed_integral_t<E>> countl_one(
    abi_tag tag, simd<E> val) noexcept {
    using sbit = to_signed_integral_t<E>;
    auto vval = +xmm::reinterpret<sbit>(tag, val);
    return xmm::countl_zero(
        tag, simd<sbit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}
#endif

template <simd_element E>
requires (sizeof(E) == 2)
inline simd<to_signed_integral_t<E>> countl_zero(
    abi_tag tag, simd<E> val) noexcept {
    auto vval = +xmm::reinterpret<to_signed_integral_t<E>>(tag, val);
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 1));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 2));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 4));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 8));
    return _mm_sub_epi16(_mm_set1_epi16(16), xmm::popcount(tag, vval));
}

template <simd_element E>
requires (sizeof(E) == 1)
inline simd<to_signed_integral_t<E>> countl_zero(
    abi_tag tag, simd<E> val) noexcept {
    auto vval = +xmm::reinterpret<to_signed_integral_t<E>>(tag, val);
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 1));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 2));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 4));
    return _mm_sub_epi8(_mm_set1_epi8(8), xmm::popcount(tag, vval));
}

template <simd_element E>
requires (sizeof(E) <= 2)
inline simd<to_signed_integral_t<E>> countl_one(
    abi_tag tag, simd<E> val) noexcept {
    using sbit = to_signed_integral_t<E>;
    auto vval = +xmm::reinterpret<sbit>(tag, val);
    return xmm::countl_zero(
        tag, simd<sbit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E>
inline simd<to_signed_integral_t<E>> countr_zero(
    abi_tag tag, simd<E> val) noexcept {
    if constexpr (!unsigned_integral<E>) {
        using bit = to_unsigned_integral_t<E>;
        return xmm::countr_zero(tag, xmm::reinterpret<bit>(tag, val));
    } else {
        auto const nval = xmm::sub(tag, simd<E>(_mm_setzero_si128()), val);
        return xmm::popcount(tag,
            xmm::sub(tag, simd<E>(_mm_and_si128(+val, +nval)),
                xmm::broadcast<E>(tag, 1u)));
    }
}

template <simd_element E>
inline simd<to_signed_integral_t<E>> countr_one(
    abi_tag tag, simd<E> val) noexcept {
    using bit = to_unsigned_integral_t<E>;
    auto vval = +xmm::reinterpret<bit>(tag, val);
    return xmm::countr_zero(
        tag, simd<bit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
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
inline simd<E> byteswap(abi_tag tag, simd<E> val) noexcept {
    if constexpr (!same_as<native_vector_t<E>, __m128i>) {
        using bit = to_unsigned_integral_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::byteswap(tag, xmm::reinterpret<bit>(tag, val)));
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

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
