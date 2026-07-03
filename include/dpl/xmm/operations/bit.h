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

#  include "dpl/std/bit/countl.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/broadcast.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

namespace details {
[[maybe_unused]] alignas(16) inline constexpr char popcnt_lut[]{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL popcount(simd<E> val) noexcept {
    if constexpr (sizeof(E) == 1) {
#if DPL_SIMD_X86_AVX512BITALG && DPL_SIMD_X86_AVX512VL
        return _mm_popcnt_epi8(
            +xmm::reinterpret<signed_representation_t<E>>(val));
#else
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::popcnt_lut));
        auto const mask = _mm_set1_epi8(0x0f);
        auto const vval = +xmm::reinterpret<int8>(val);
        auto const lo = _mm_and_si128(vval, mask);
        auto const hi = _mm_and_si128(_mm_srli_epi16(vval, 4), mask);
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

#if DPL_SIMD_X86_AVX512VL
#  if DPL_SIMD_X86_AVX512VPOPCNTDQ
template <imask_t<uint32> M, common_size_with<uint32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint32>
    DPL_VECTORCALL popcount(
        simd<uint32> src, cmask_t<uint32, M>, simd<E> val) noexcept {
    return _mm_mask_popcnt_epi32(+src, M, +xmm::reinterpret<uint32>(val));
}

template <imask_t<uint32> M, common_size_with<uint32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint32>
    DPL_VECTORCALL popcount(
        dx::zero_t, cmask_t<uint32, M>, simd<E> val) noexcept {
    return _mm_maskz_popcnt_epi32(M, +xmm::reinterpret<uint32>(val));
}

template <imask_t<uint64> M, common_size_with<uint64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint64>
    DPL_VECTORCALL popcount(
        simd<uint64> src, cmask_t<uint64, M>, simd<E> val) noexcept {
    return _mm_mask_popcnt_epi64(+src, M, +xmm::reinterpret<uint64>(val));
}

template <imask_t<uint64> M, common_size_with<uint64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint64>
    DPL_VECTORCALL popcount(
        dx::zero_t, cmask_t<uint64, M>, simd<E> val) noexcept {
    return _mm_maskz_popcnt_epi64(M, +xmm::reinterpret<uint64>(val));
}
#  endif

#  if DPL_SIMD_X86_AVX512BITALG
template <imask_t<uint16> M, common_size_with<uint16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint16>
    DPL_VECTORCALL popcount(
        simd<uint16> src, cmask_t<uint16, M>, simd<E> val) noexcept {
    return _mm_mask_popcnt_epi16(+src, M, +xmm::reinterpret<uint16>(val));
}

template <imask_t<uint16> M, common_size_with<uint16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint16>
    DPL_VECTORCALL popcount(
        dx::zero_t, cmask_t<uint16, M>, simd<E> val) noexcept {
    return _mm_maskz_popcnt_epi16(M, +xmm::reinterpret<uint16>(val));
}

template <imask_t<uint8> M, common_size_with<uint8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint8>
    DPL_VECTORCALL popcount(
        simd<uint8> src, cmask_t<uint8, M>, simd<E> val) noexcept {
    return _mm_mask_popcnt_epi8(+src, M, +xmm::reinterpret<uint8>(val));
}

template <imask_t<uint8> M, common_size_with<uint8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint8>
    DPL_VECTORCALL popcount(
        dx::zero_t, cmask_t<uint8, M>, simd<E> val) noexcept {
    return _mm_maskz_popcnt_epi8(M, +xmm::reinterpret<uint8>(val));
}
#  endif
#endif

template <common_size_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
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

template <common_size_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
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

#if DPL_SIMD_X86_AVX512CD && DPL_SIMD_X86_AVX512VL
template <imask_t<uint64> M, common_size_with<uint64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint64>
    DPL_VECTORCALL countl_zero(
        simd<uint64> src, cmask_t<E, M>, simd<E> val) noexcept {
    return _mm_mask_lzcnt_epi64(+src, M, +xmm::reinterpret<uint64>(val));
}

template <imask_t<uint64> M, common_size_with<uint64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint64>
    DPL_VECTORCALL countl_zero(
        dx::zero_t, cmask_t<E, M>, simd<E> val) noexcept {
    return _mm_maskz_lzcnt_epi64(M, +xmm::reinterpret<uint64>(val));
}

template <imask_t<uint32> M, common_size_with<uint32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint32>
    DPL_VECTORCALL countl_zero(
        simd<uint32> src, cmask_t<E, M>, simd<E> val) noexcept {
    return _mm_mask_lzcnt_epi32(+src, M, +xmm::reinterpret<uint32>(val));
}

template <imask_t<uint32> M, common_size_with<uint32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint32>
    DPL_VECTORCALL countl_zero(
        dx::zero_t, cmask_t<E, M>, simd<E> val) noexcept {
    return _mm_maskz_lzcnt_epi32(M, +xmm::reinterpret<uint32>(val));
}
#endif

template <common_size_with<uint16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint16>
    DPL_VECTORCALL countl_zero(simd<E> val) noexcept {
    auto vval = +xmm::reinterpret<uint16>(val);
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 1));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 2));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 4));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 8));
    return _mm_sub_epi16(_mm_set1_epi16(16), xmm::popcount(vval));
}

template <common_size_with<uint8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<uint8>
    DPL_VECTORCALL countl_zero(simd<E> val) noexcept {
    auto vval = +xmm::reinterpret<uint8>(val);
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 1));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 2));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 4));
    return _mm_sub_epi8(_mm_set1_epi8(8), xmm::popcount(vval));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL countl_one(simd<E> val) noexcept {
    using sbit = signed_representation_t<E>;
    auto vval = +xmm::reinterpret<sbit>(val);
    return xmm::countl_zero(
        simd<sbit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL countl_one(simd<unsigned_representation_t<E>> src,
        cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::countl_zero(src, mask, src); }
{
    using ubit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<ubit>(val);
    return xmm::countl_zero(src, mask,
        simd<ubit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL countl_one(
        dx::zero_t zero, cmask_t<E, M> mask, simd<E> val) noexcept
requires requires(simd<unsigned_representation_t<E>> rep) {
    xmm::countl_zero(zero, mask, rep);
}
{
    using ubit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<ubit>(val);
    return xmm::countl_zero(zero, mask,
        simd<ubit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
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

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_zero(simd<unsigned_representation_t<E>> src,
        cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::popcount(src, mask, src); }
{
    if constexpr (!unsigned_integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::countr_zero(src, mask, xmm::reinterpret<bit>(val));
    } else {
        auto const nval = xmm::subtract(simd<E>(_mm_setzero_si128()), val);
        return xmm::popcount(src, mask,
            xmm::subtract(
                simd<E>(_mm_and_si128(+val, +nval)), xmm::broadcast<E>(1u)));
    }
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_zero(
        dx::zero_t zero, cmask_t<E, M> mask, simd<E> val) noexcept
requires requires(
    simd<unsigned_representation_t<E>> src) { xmm::popcount(zero, mask, src); }
{
    if constexpr (!unsigned_integral<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::countr_zero(zero, mask, xmm::reinterpret<bit>(val));
    } else {
        auto const nval = xmm::subtract(simd<E>(_mm_setzero_si128()), val);
        return xmm::popcount(zero, mask,
            xmm::subtract(
                simd<E>(_mm_and_si128(+val, +nval)), xmm::broadcast<E>(1u)));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_one(simd<E> val) noexcept {
    using bit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<bit>(val);
    return xmm::countr_zero(
        simd<bit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_one(simd<unsigned_representation_t<E>> src,
        cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::countr_zero(src, mask, src); }
{
    using ubit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<ubit>(val);
    return xmm::countr_zero(src, mask,
        simd<ubit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<unsigned_representation_t<E>>
    DPL_VECTORCALL countr_one(
        dx::zero_t zero, cmask_t<E, M> mask, simd<E> val) noexcept
requires requires(simd<unsigned_representation_t<E>> rep) {
    xmm::countr_zero(zero, mask, rep);
}
{
    using ubit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<ubit>(val);
    return xmm::countr_zero(zero, mask,
        simd<ubit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
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

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL popcount(abi_tag, type_identity_t<simd<E>> src,
    cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::popcount(src, mask, val); }
{
    return xmm::popcount(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_zero(abi_tag, type_identity_t<simd<E>> src,
    cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::countl_zero(src, mask, val); }
{
    return xmm::countl_zero(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_zero(abi_tag, type_identity_t<simd<E>> src,
    cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::countl_zero(src, mask, val); }
{
    return xmm::countl_zero(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL popcount(
    abi_tag, dx::zero_t zero, cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::popcount(zero, mask, val); }
{
    return xmm::popcount(zero, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_zero(
    abi_tag, dx::zero_t zero, cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::countl_zero(zero, mask, val); }
{
    return xmm::countl_zero(zero, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countr_zero(
    abi_tag, dx::zero_t zero, cmask_t<E, M> mask, simd<E> val) noexcept
requires requires { xmm::countl_zero(zero, mask, val); }
{
    return xmm::countl_zero(zero, mask, val);
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
