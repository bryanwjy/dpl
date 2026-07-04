// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/operations/arithmetic.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/arithmetic/fwd.h"

#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/std/utility/template_barrier.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_EXPORT template <template_barrier_t = template_barrier, simd_element E>
requires integral<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(simd<E> lhs, simd<E> rhs) noexcept {
    if constexpr (sizeof(E) == sizeof(int64)) {
#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
        return _mm_mullo_epi64(+lhs, +rhs);
#else
        auto const vlhs = +lhs;
        auto const vrhs = +rhs;
        auto const left_lo = vlhs;
        auto const right_lo = vrhs;
        auto const left_hi = _mm_srli_epi64(vlhs, 32);
        auto const right_hi = _mm_srli_epi64(vrhs, 32);
        auto const lo_lo = _mm_mul_epu32(left_lo, right_lo);
        auto const lo_hi = _mm_mul_epu32(left_lo, right_hi);
        auto const hi_lo = _mm_mul_epu32(left_hi, right_lo);
        return _mm_add_epi64(
            lo_lo, _mm_slli_epi64(_mm_add_epi64(lo_hi, hi_lo), 32));
#endif
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_mullo_epi32(+lhs, +rhs);
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_mullo_epi16(+lhs, +rhs);
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const zero = _mm_setzero_si128();
        auto const vlhs = +lhs;
        auto const vrhs = +rhs;
        static constexpr auto unpacklo = [](__m128i val, __m128i zero) {
            if constexpr (unsigned_integral<E>) {
                return _mm_unpacklo_epi8(val, zero);
            } else {
                return _mm_unpacklo_epi8(
                    val, _mm_cmpgt_epi8(_mm_setzero_si128(), val));
            }
        };
        static constexpr auto unpackhi = [](__m128i val, __m128i zero) {
            if constexpr (unsigned_integral<E>) {
                return _mm_unpackhi_epi8(val, zero);
            } else {
                return _mm_unpackhi_epi8(
                    val, _mm_cmpgt_epi8(_mm_setzero_si128(), val));
            }
        };
        auto const left_lo = unpacklo(vlhs, zero);
        auto const left_hi = unpackhi(vlhs, zero);
        auto const right_lo = unpacklo(vrhs, zero);
        auto const right_hi = unpackhi(vrhs, zero);

        auto const prod_lo = _mm_mullo_epi16(left_lo, right_lo);
        auto const prod_hi = _mm_mullo_epi16(left_hi, right_hi);
        return _mm_packus_epi16(prod_lo, prod_hi);
    }
}

DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<float>
    DPL_VECTORCALL multiply(simd<float> lhs, simd<float> rhs) noexcept {
    return _mm_mul_ps(+lhs, +rhs);
}

DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<double>
    DPL_VECTORCALL multiply(simd<double> lhs, simd<double> rhs) noexcept {
    return _mm_mul_pd(+lhs, +rhs);
}

#if DPL_SIMD_X86_AVX512VL
#  if DPL_SIMD_X86_AVX512F

DPL_EXPORT template <imask_t<int32> M, integral E>
requires common_size_with<E, int32>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(type_identity_t<simd<E>> src, cmask_t<E, M>,
        simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mask_mullo_epi32(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int32> M, integral E>
requires common_size_with<E, int32>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(
        dx::zero_t, cmask_t<E, M>, simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_maskz_mullo_epi32(M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<float>
    DPL_VECTORCALL multiply(simd<float> src, cmask_t<float, M>, simd<float> lhs,
        simd<float> rhs) noexcept {
    return _mm_mask_mul_ps(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<double>
    DPL_VECTORCALL multiply(simd<double> src, cmask_t<double, M>,
        simd<double> lhs, simd<double> rhs) noexcept {
    return _mm_mask_mul_pd(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<float>
    DPL_VECTORCALL multiply(dx::zero_t, cmask_t<float, M>, simd<float> lhs,
        simd<float> rhs) noexcept {
    return _mm_maskz_mul_ps(M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<double>
    DPL_VECTORCALL multiply(dx::zero_t, cmask_t<double, M>, simd<double> lhs,
        simd<double> rhs) noexcept {
    return _mm_maskz_mul_pd(M, +lhs, +rhs);
}

#  endif

#  if DPL_SIMD_X86_AVX512DQ

DPL_EXPORT template <imask_t<int64> M, integral E>
requires common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(type_identity_t<simd<E>> src, cmask_t<E, M>,
        simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mask_mullo_epi64(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int64> M, integral E>
requires common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(
        dx::zero_t, cmask_t<E, M>, simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_maskz_mullo_epi64(M, +lhs, +rhs);
}
#  endif

#  if DPL_SIMD_X86_AVX512BW

DPL_EXPORT template <imask_t<int16> M, integral E>
requires common_size_with<E, int16>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(type_identity_t<simd<E>> src, cmask_t<E, M>,
        simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mask_mullo_epi16(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int16> M, integral E>
requires common_size_with<E, int16>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(
        dx::zero_t, cmask_t<E, M>, simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_maskz_mullo_epi16(M, +lhs, +rhs);
}
#  endif

#  if DPL_SIMD_X86_AVX512FP16

DPL_EXPORT template <simd_element E>
requires float16_like<representation_t<E>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mul_ph(+lhs, +rhs);
}

DPL_EXPORT template <imask_t<int16> M, simd_element E>
requires float16_like<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(type_identity_t<simd<E>> src, cmask_t<E, M>,
        simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mask_mul_ph(src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int16> M, simd_element E>
requires float16_like<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(
        dx::zero_t zero, cmask_t<E, M>, simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_maskz_mul_ph(M, +lhs, +rhs);
}

#  endif
#endif

DPL_EXPORT template <simd_element E>
requires bfloat16_like<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(simd<E> lhs, simd<E> rhs) noexcept {

    auto const hlhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo = xmm::multiply(
        xmm::element_cast<float>(lhs), xmm::element_cast<float>(rhs));
    auto const hi = xmm::multiply(
        xmm::element_cast<float>(hlhs), xmm::element_cast<float>(hrhs));

#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(hi, lo);
#else
    using sbit = signed_representation_t<E>;
    auto const packed =
        _mm_packus_epi32(__DPL bit_cast<__m128i>(+xmm::element_cast<E>(lo)),
            __DPL bit_cast<__m128i>(+xmm::element_cast<E>(hi)));

    return xmm::reinterpret<E>(simd<sbit>(packed));
#endif
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL multiply(abi_tag, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::multiply(lhs, rhs); }
{
    return xmm::multiply(lhs, rhs);
}

DPL_EXPORT template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(abi_tag, type_identity_t<simd<E>> src,
        cmask_t<E, M> mask, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::multiply(src, mask, lhs, rhs); }
{
    return xmm::multiply(src, mask, lhs, rhs);
}

DPL_EXPORT template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
        simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::multiply(zero, mask, lhs, rhs); }
{
    return xmm::multiply(zero, mask, lhs, rhs);
}
} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
